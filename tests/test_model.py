"""build_dummy_model/build_dummy_input 테스트.

이 환경에는 transformers/peft/bitsandbytes/CUDA가 없다(pyproject.toml이 의도적으로
하드 의존성에 넣지 않음 — docs/architecture.md §2 참고, CUDA는 이 dev 머신 자체에 없음).
그래서 두 층으로 나눠 검증한다.

1. **구조/로직 검증(이 파일의 핵심)** — meta 디바이스 골격 → 레이어 단위 실체화 →
   LoRA 훅 부착이라는 알고리즘 자체가 맞는지는, `bitsandbytes.nn.Linear4bit`/
   `Params4bit`를 최소한의 진짜 torch 서브클래스로 흉내 내서 **실제 torch 연산**으로
   검증한다(가짜 텐서 연산이 아니라 진짜 forward/backward가 돈다). device="cpu"로
   돌려서 CUDA 없이도 실행 가능하게 한다.
2. **bitsandbytes 자체의 4bit 양자화 정확성**은 이 테스트가 검증하지 않는다 —
   상영님이 RTX 4070 Ti에서 실물 라이브러리로 실측 검증했다(PR #12 리뷰 코멘트,
   2026-08-06). 여기서는 그 검증된 알고리즘을 내가 정확히 옮겨 적었는지만 본다.
3. **`@pytest.mark.network`로 표시한 실물 통합 테스트 하나**는 모킹을 전혀 안 쓰고
   실제 `transformers`/`bitsandbytes`/HuggingFace Hub로 돈다(상영님 제안,
   2026-08-06) — 모킹 테스트가 "논리적으로 그럴듯한 조합"만 검증하고 "실제로
   맞물리는지"는 검증 못 해서 QLoRA 구현이 통째로 죽어있던 걸 못 잡았던 문제의
   재발 방지용이다. 기본 실행(`pytest`)에서는 빠지고 `pytest -m network`로만
   켠다 — `run_canary_check()`가 아니라 `build_dummy_model()`을 직접 호출한다,
   `--model` 경로는 아직 worker.py에 배선되지 않아서(W9 대기, canary/worker.py
   참고) `run_canary_check()`를 쓰면 status가 항상 "error"로 나온다.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch
from torch import nn


class _FakeConfig:
    def __init__(self, vocab_size: int = 32000) -> None:
        self.vocab_size = vocab_size


class _FakeParams4bit(nn.Parameter):
    """bitsandbytes.nn.Params4bit을 흉내 낸다 — 진짜 nn.Parameter 서브클래스라야
    setattr(module, "weight", ...)로 대입했을 때 model.parameters()에 잡힌다."""

    def __new__(cls, data=None, requires_grad: bool = False, quant_type: str = "nf4"):
        instance = torch.Tensor._make_subclass(cls, data, requires_grad)
        instance.quant_type = quant_type
        return instance

    def to(self, *args, **kwargs):
        moved = super().to(*args, **kwargs)
        result = _FakeParams4bit(
            moved.data, requires_grad=self.requires_grad, quant_type=self.quant_type
        )
        return result


class _FakeLinear4bit(nn.Module):
    """bitsandbytes.nn.Linear4bit을 흉내 낸다 — meta 파라미터를 가진 자리표시자.

    실제 4bit 연산은 하지 않고(양자화 정확성은 상영님이 실물로 검증), forward가
    shape만 맞춰 진짜 텐서 연산 그래프를 이어준다 — 그래야 LoRA 훅의 backward가
    실제로 검증된다.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device="meta"), requires_grad=False
        )

    def forward(self, x):
        weight = self.weight
        if weight.device.type == "meta":
            raise RuntimeError("Linear4bit weight가 아직 meta 상태 — 실체화 전에 forward 호출됨")
        return x @ weight.to(x.dtype).t()


def _fake_replace_with_bnb_linear(model, modules_to_not_convert=None, quantization_config=None):
    """실제 replace_with_bnb_linear를 흉내 — nn.Linear를 _FakeLinear4bit로 교체한다."""
    modules_to_not_convert = modules_to_not_convert or []
    for name, child in list(model.named_children()):
        if name in modules_to_not_convert:
            continue
        if isinstance(child, nn.Linear):
            replacement = _FakeLinear4bit(child.in_features, child.out_features)
            setattr(model, name, replacement)
        else:
            _fake_replace_with_bnb_linear(child, modules_to_not_convert, quantization_config)
    return model


class _TinyMetaModel(nn.Module):
    """AutoModelForCausalLM.from_config()이 meta 디바이스 위에서 만들어줄 법한
    아주 작은 골격 — Linear 2개 + lm_head(변환 제외 대상)."""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(8, 8, bias=False)
        self.layer2 = nn.Linear(8, 8, bias=False)
        self.lm_head = nn.Linear(8, 16, bias=False)

    def forward(self, x):
        return self.lm_head(self.layer2(self.layer1(x)))


@pytest.fixture
def fake_transformers(monkeypatch):
    """AutoConfig.from_pretrained · AutoModelForCausalLM.from_config(meta 지원) ·
    BitsAndBytesConfig · transformers.integrations.bitsandbytes.replace_with_bnb_linear
    를 흉내 낸다."""
    calls = {}

    class _FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_name):
            calls["config_model_name"] = model_name
            return _FakeConfig()

    class _FakeAutoModelForCausalLM:
        @staticmethod
        def from_config(config, dtype=None):
            # 실제로는 quantization_config kwarg를 받지 않는다(v1 버그의 원인이었던
            # 부분) — 이 fake도 quantization_config를 받지 않게 시그니처를 맞춰서,
            # 코드가 실수로 다시 그 kwarg를 넘기면 TypeError로 테스트가 잡아준다.
            calls["from_config_dtype"] = dtype
            return _TinyMetaModel()

    class _FakeBitsAndBytesConfig:
        def __init__(self, **kwargs) -> None:
            calls["bnb_config_kwargs"] = kwargs

    fake_transformers_module = types.SimpleNamespace(
        AutoConfig=_FakeAutoConfig,
        AutoModelForCausalLM=_FakeAutoModelForCausalLM,
        BitsAndBytesConfig=_FakeBitsAndBytesConfig,
    )
    fake_integrations_bnb = types.SimpleNamespace(
        replace_with_bnb_linear=_fake_replace_with_bnb_linear
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers_module)
    monkeypatch.setitem(sys.modules, "transformers.integrations", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "transformers.integrations.bitsandbytes", fake_integrations_bnb
    )
    return calls


@pytest.fixture
def fake_bitsandbytes(monkeypatch):
    """bitsandbytes.nn.Linear4bit/Params4bit을 흉내 낸다."""
    fake_bnb_nn = types.SimpleNamespace(Linear4bit=_FakeLinear4bit, Params4bit=_FakeParams4bit)
    fake_bnb = types.SimpleNamespace(nn=fake_bnb_nn)
    monkeypatch.setitem(sys.modules, "bitsandbytes", fake_bnb)
    monkeypatch.setitem(sys.modules, "bitsandbytes.nn", fake_bnb_nn)
    return fake_bnb


def _remove_module(monkeypatch, name: str) -> None:
    """name이 실제로 설치돼 있어도(이 환경엔 없지만) 없는 것처럼 만든다."""
    monkeypatch.delitem(sys.modules, name, raising=False)
    real_import = __import__

    def _blocking_import(module_name, *args, **kwargs):
        if module_name == name or module_name.startswith(name + "."):
            raise ModuleNotFoundError(f"No module named {module_name!r} (blocked for test)")
        return real_import(module_name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _blocking_import)


def test_build_dummy_model_uses_config_only_no_weights(
    fake_transformers, fake_bitsandbytes
) -> None:
    from preflight.canary.model import build_dummy_model

    _model, _config, quant_backend = build_dummy_model("some-org/some-model", device="cpu")

    assert fake_transformers["config_model_name"] == "some-org/some-model"
    assert quant_backend == "bnb-4bit"


def test_build_dummy_model_builds_on_meta_device_first(
    fake_transformers, fake_bitsandbytes
) -> None:
    """AutoModelForCausalLM.from_config()이 meta 디바이스에서 호출됐는지 확인한다.

    dtype=torch.float16으로 호출돼야 한다(상영님 검증 스니펫과 동일).
    """
    from preflight.canary.model import build_dummy_model

    build_dummy_model("some-org/some-model", device="cpu")

    assert fake_transformers["from_config_dtype"] == torch.float16


def test_build_dummy_model_no_meta_tensors_left_after_materialization(
    fake_transformers, fake_bitsandbytes
) -> None:
    """실체화 루프가 끝나면 meta 디바이스에 남은 파라미터가 하나도 없어야 한다."""
    from preflight.canary.model import build_dummy_model

    model, _config, _quant_backend = build_dummy_model("some-org/some-model", device="cpu")

    meta_params = [name for name, p in model.named_parameters() if p.device.type == "meta"]
    assert meta_params == []


def test_build_dummy_model_freezes_base_and_trains_only_lora(
    fake_transformers, fake_bitsandbytes
) -> None:
    """베이스(Linear4bit 가중치)는 얼리고, LoRA 어댑터(lora_down/lora_up)만 학습 대상이다."""
    from preflight.canary.model import build_dummy_model

    model, _config, _quant_backend = build_dummy_model("some-org/some-model", device="cpu")

    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}

    assert trainable, "LoRA 어댑터가 model.parameters()에서 학습 대상으로 잡혀야 한다"
    assert all("lora_down" in name or "lora_up" in name for name in trainable)
    assert any("weight" in name and "lora" not in name for name in frozen)


def test_build_dummy_model_lora_hook_actually_changes_output_and_gets_gradient(
    fake_transformers, fake_bitsandbytes
) -> None:
    """LoRA 훅이 실제로 forward 출력에 관여하고, backward 시 어댑터에 gradient가 흐른다.

    up을 0으로 초기화하는 관례상 첫 forward는 베이스와 동일한 값이 나온다 —
    그래서 up.weight를 직접 흔들어 훅이 실제로 출력에 반영되는지 확인한다.
    """
    from preflight.canary.model import build_dummy_model

    model, _config, _quant_backend = build_dummy_model("some-org/some-model", device="cpu")
    # 실체화 루프가 항상 float16으로 만들기 때문에(compute_dtype=torch.float16과
    # 일치시키기 위함) 입력도 맞춰준다 — 실제 파이프라인에서는 임베딩 레이어 출력이
    # 이미 float16이라 자연히 맞는다.
    dummy_input = torch.randn(2, 8, dtype=torch.float16)

    baseline = model(dummy_input)

    lora_up_params = [p for name, p in model.named_parameters() if name.endswith("lora_up.weight")]
    assert lora_up_params
    with torch.no_grad():
        lora_up_params[0].add_(1.0)

    perturbed = model(dummy_input)
    assert not torch.allclose(baseline, perturbed), (
        "lora_up을 흔들었는데 출력이 그대로면 훅이 안 걸린 것"
    )

    loss = model(dummy_input).sum()
    loss.backward()
    assert lora_up_params[0].grad is not None
    assert (lora_up_params[0].grad != 0).any()


def test_build_dummy_model_falls_back_to_fp32_when_bitsandbytes_missing(
    fake_transformers, monkeypatch
) -> None:
    """bitsandbytes가 없으면(이 환경의 실제 상태) 죽지 않고 fp32 전체 모델로 폴백한다."""
    _remove_module(monkeypatch, "bitsandbytes")

    from preflight.canary.model import build_dummy_model

    _model, _config, quant_backend = build_dummy_model("some-org/some-model", device="cpu")

    assert quant_backend == "nn-linear-fallback"


def test_build_dummy_input_uses_vocab_size_from_config() -> None:
    from preflight.canary.model import build_dummy_input

    vocab_size = 128
    input_ids = build_dummy_input(batch_size=2, seq_len=6, vocab_size=vocab_size)

    assert input_ids.shape == (2, 6)
    assert input_ids.dtype in (torch.int64, torch.int32)
    assert int(input_ids.min()) >= 0
    assert int(input_ids.max()) < vocab_size


def test_build_dummy_model_and_input_are_compatible(fake_transformers, fake_bitsandbytes) -> None:
    """build_dummy_model이 돌려준 config.vocab_size로 build_dummy_input을 바로 쓸 수 있다."""
    from preflight.canary.model import build_dummy_input, build_dummy_model

    _model, config, _quant_backend = build_dummy_model("some-org/some-model", device="cpu")
    input_ids = build_dummy_input(batch_size=1, seq_len=4, vocab_size=config.vocab_size)

    assert input_ids.shape == (1, 4)


@pytest.mark.network
def test_model_path_actually_applies_4bit_with_real_libraries() -> None:
    """모킹 없이 실물 transformers/bitsandbytes + HuggingFace Hub로 QLoRA가 진짜 걸리는지 확인한다.

    상영님 제안(PR #12 리뷰, 2026-08-06) — `hf-internal-testing/tiny-random-
    LlamaForCausalLM`은 config.json 수 KB만 받고 가중치는 안 받는다(architecture.md
    §5 "가중치는 다운로드하지 않는다"와 여전히 일치). Llama 계열이라 실제 타깃
    모델과 구조가 같다.

    기본 `pytest`에서는 빠진다 — `pytest -m network`로 명시적으로 켠다
    (pyproject.toml [tool.pytest.ini_options] 참고). CI가 아니라 로컬에서
    transformers/bitsandbytes 실물 라이브러리와 GPU를 가진 사람이 돌리는 걸
    전제로 한다 — 이 저장소의 dev 환경(CI 포함)엔 그 라이브러리들이 없어서
    이 테스트를 실행할 수 없다(하드 의존성이 아니므로 의도된 상태, docs/
    architecture.md §2 참고). 상영님 쪽 환경에서 이 테스트가 실제로 통과하는지
    확인 부탁드립니다 — 제가 검증할 방법이 없습니다.
    """
    from preflight.canary.model import build_dummy_model

    model, _config, quant_backend = build_dummy_model(
        "hf-internal-testing/tiny-random-LlamaForCausalLM", device="cuda"
    )

    assert quant_backend == "bnb-4bit"

    import bitsandbytes as bnb

    linear4bit_count = sum(1 for m in model.modules() if isinstance(m, bnb.nn.Linear4bit))
    assert linear4bit_count > 0, "Linear4bit 레이어가 하나도 없다 — 양자화가 실제로 안 걸린 것"
