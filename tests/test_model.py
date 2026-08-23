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

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")


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
        # 실물 RoPE의 `inv_freq`와 같은 성질의 버퍼 — 학습 대상이 아니고
        # persistent=False다. `with torch.device("meta")` 안에서 만들어지면
        # 파라미터와 똑같이 meta로 가는데, 실체화 루프가 파라미터만 훑으면
        # 여기 남아 forward에서 죽는다(#134).
        self.register_buffer("inv_freq", torch.ones(8), persistent=False)

    def forward(self, x):
        # 버퍼를 실제로 쓴다 — 안 쓰면 meta로 남아 있어도 forward가 통과해버려
        # #134가 또 샌다. **더하기**인 것은 실물을 흉내 낸 것이다: 실체화 값이 0일 때
        # RoPE는 `cos=1, sin=0`이라 항등이 되는데, 곱하기로 쓰면 입력이 통째로
        # 0이 돼 다른 테스트(LoRA 훅이 출력을 바꾸는지)까지 무의미해진다.
        # `.to(x.dtype)`도 실물과 같다 — 실제 `inv_freq`는 float32인데 모델은
        # float16이라, RoPE가 쓰는 쪽에서 맞춰준다.
        return self.lm_head(self.layer2(self.layer1(x + self.inv_freq.to(x.dtype))))


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
    """bitsandbytes가 없으면 죽지 않고 양자화 없는 nn.Linear 베이스로 폴백한다."""
    _remove_module(monkeypatch, "bitsandbytes")

    from preflight.canary.model import build_dummy_model

    _model, _config, quant_backend = build_dummy_model("some-org/some-model", device="cpu")

    assert quant_backend == "nn-linear-fallback"


def test_build_dummy_model_fallback_moves_model_to_requested_device(
    fake_transformers, monkeypatch
) -> None:
    """폴백 모델도 요청받은 device로 올라간다 (#66).

    이걸 안 하면 폴백 모델만 CPU에 남아, 같은 device로 만들어진 입력(cuda)과
    어긋나 forward가 `RuntimeError: Expected all tensors to be on the same
    device`로 죽는다 — "bitsandbytes가 없어도 진단은 계속한다"는 폴백의 목적이
    발동 순간 무너진다.

    `device="meta"`로 검증한다 — GPU 없는 CI에서도 "요청 device로 옮겼는가"를
    실제 파라미터 device로 확인할 수 있고, 실제 메모리는 할당하지 않는다.
    """
    _remove_module(monkeypatch, "bitsandbytes")

    from preflight.canary.model import build_dummy_model

    model, _config, quant_backend = build_dummy_model("some-org/some-model", device="meta")

    assert quant_backend == "nn-linear-fallback"
    param_devices = {param.device.type for param in model.parameters()}
    assert param_devices == {"meta"}, f"폴백 모델이 요청 device로 안 올라갔다: {param_devices}"


# ── 폴백도 QLoRA의 모양을 유지한다 (#75) ──────────────────────────────────────
#
# 4bit 구성이 실패해도 "베이스는 얼고 어댑터만 학습한다"는 부분은 torch만으로 된다.
# 안 하면 폴백이 fp32 전체 파인튜닝이 되어 파라미터당 16바이트(가중치+gradient+
# AdamW 상태)가 들고, canary가 재는 대상이 사용자의 계획(QLoRA)과 달라진다.


def test_build_dummy_model_fallback_freezes_base_and_trains_only_lora(
    fake_transformers, monkeypatch
) -> None:
    """폴백 모델도 베이스는 얼고 LoRA 어댑터만 학습 대상이다 (#75).

    4bit 경로의 동일 테스트(test_build_dummy_model_freezes_base_and_trains_only_lora)와
    대칭이다 — 양자화만 빠지고 학습 경로의 모양은 같아야 한다.
    """
    _remove_module(monkeypatch, "bitsandbytes")

    from preflight.canary.model import build_dummy_model

    model, _config, quant_backend = build_dummy_model("some-org/some-model", device="cpu")

    assert quant_backend == "nn-linear-fallback"
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}

    assert trainable, "LoRA 어댑터가 학습 대상으로 잡혀야 한다"
    assert all("lora_down" in name or "lora_up" in name for name in trainable), trainable
    assert "layer1.weight" in frozen and "layer2.weight" in frozen, frozen


def test_build_dummy_model_fallback_skips_lm_head_like_the_4bit_path(
    fake_transformers, monkeypatch
) -> None:
    """폴백의 어댑터 대상은 4bit 경로의 변환 대상과 같다 — lm_head는 얼기만 한다.

    두 경로가 "베이스로 취급하는 레이어"를 다르게 잡으면, 폴백이 재는 대상이
    4bit 경로와 달라져 비교 자체가 성립하지 않는다.
    """
    _remove_module(monkeypatch, "bitsandbytes")

    from preflight.canary.model import build_dummy_model

    model, _config, _quant_backend = build_dummy_model("some-org/some-model", device="cpu")

    names = dict(model.named_parameters())
    assert "lm_head.lora_down.weight" not in names
    assert names["lm_head.weight"].requires_grad is False


def test_build_dummy_model_fallback_lora_hook_changes_output_and_gets_gradient(
    fake_transformers, monkeypatch
) -> None:
    """폴백 LoRA 훅이 실제 forward에 관여하고 backward에서 gradient를 받는다.

    up을 0으로 초기화하는 관례상 첫 forward는 베이스와 같은 값이라, up.weight를
    직접 흔들어 훅이 출력에 반영되는지 확인한다(4bit 경로 테스트와 동일한 방법).
    """
    _remove_module(monkeypatch, "bitsandbytes")

    from preflight.canary.model import build_dummy_model

    model, _config, _quant_backend = build_dummy_model("some-org/some-model", device="cpu")
    # 폴백 모델은 dtype 지정 없이 from_config로 만들어져 float32다 — 어댑터도 베이스
    # weight의 dtype을 따라가므로 입력만 맞춰주면 된다.
    dummy_input = torch.randn(2, 8)

    baseline = model(dummy_input)

    lora_up = [p for name, p in model.named_parameters() if name.endswith("lora_up.weight")]
    assert lora_up
    with torch.no_grad():
        lora_up[0].add_(1.0)

    perturbed = model(dummy_input)
    assert not torch.allclose(baseline, perturbed), (
        "lora_up을 흔들었는데 출력이 그대로면 훅이 안 걸린 것"
    )

    model(dummy_input).sum().backward()
    assert lora_up[0].grad is not None
    assert (lora_up[0].grad != 0).any()


def test_freeze_base_and_attach_lora_cuts_trainable_params_to_a_small_fraction() -> None:
    """학습 대상이 어댑터로 줄어드는 것을 수치로 확인한다 (#75의 요지).

    gradient·AdamW 상태는 **학습 대상 파라미터에만** 붙으므로, 이 비율이 그대로
    폴백 실행의 메모리 절감폭이 된다.
    """
    from preflight.canary.model import _freeze_base_and_attach_lora

    hidden = 512
    model = nn.Sequential(
        nn.Linear(hidden, hidden, bias=False),
        nn.Linear(hidden, hidden, bias=False),
    )

    assert _freeze_base_and_attach_lora(model) is True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable / total < 0.1, f"학습 대상이 {trainable}/{total}로 거의 안 줄었다"


def test_freeze_base_and_attach_lora_does_not_nest_adapters_in_adapters() -> None:
    """어댑터(lora_down/lora_up)도 nn.Linear라, 순회하며 붙이면 어댑터에 또 붙는다."""
    from preflight.canary.model import _freeze_base_and_attach_lora

    model = nn.Sequential(nn.Linear(8, 8, bias=False))
    _freeze_base_and_attach_lora(model)

    nested = [name for name, _ in model.named_parameters() if name.count("lora_") > 1]
    assert nested == [], nested


def test_freeze_base_and_attach_lora_reverts_when_there_is_nothing_to_attach() -> None:
    """붙일 대상이 없으면 동결을 되돌린다 — 학습 대상 0개는 옵티마이저를 죽인다.

    `worker._execute_canary_cycle`이 `AdamW([])`를 만들면
    `ValueError: optimizer got an empty parameter list`로 죽어, 메모리를 아끼려다
    진단 자체를 잃는다.
    """
    from preflight.canary.model import _freeze_base_and_attach_lora

    class _OnlyExcludedHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(16, 4)
            self.lm_head = nn.Linear(4, 16, bias=False)

    model = _OnlyExcludedHead()

    assert _freeze_base_and_attach_lora(model) is False
    assert all(p.requires_grad for p in model.parameters())


def test_freeze_base_and_attach_lora_survives_torch_import_failure(monkeypatch) -> None:
    """`import torch`가 죽는 환경에서도 폴백을 깨뜨리지 않는다.

    이 함수를 부르는 분기는 torch import 실패로도 들어올 수 있다 — 안전장치가
    안전장치를 부수면 안 된다.
    """
    from preflight.canary.model import _freeze_base_and_attach_lora

    model = nn.Sequential(nn.Linear(8, 8, bias=False))
    _remove_module(monkeypatch, "torch")

    assert _freeze_base_and_attach_lora(model) is False
    assert all(p.requires_grad for p in model.parameters())


def test_base_layer_device_finds_frozen_base_on_fallback_model(
    fake_transformers, monkeypatch
) -> None:
    """폴백 모델도 이제 얼린 베이스를 갖는다 — worker가 첫 파라미터로 물러서지 않는다 (#66→#75)."""
    _remove_module(monkeypatch, "bitsandbytes")

    from preflight.canary.model import build_dummy_model
    from preflight.canary.worker import _base_layer_device

    model, _config, _quant_backend = build_dummy_model("some-org/some-model", device="cpu")

    frozen = [p for p in model.parameters() if not p.requires_grad]
    assert frozen, "폴백 모델에 얼린 베이스가 있어야 한다"
    assert _base_layer_device(model) == "cpu"


def test_build_dummy_input_uses_vocab_size_from_config() -> None:
    from preflight.canary.model import build_dummy_input

    vocab_size = 128
    input_ids = build_dummy_input(batch_size=2, seq_len=6, vocab_size=vocab_size, device="cpu")

    assert input_ids.shape == (2, 6)
    assert input_ids.dtype in (torch.int64, torch.int32)
    assert int(input_ids.min()) >= 0
    assert int(input_ids.max()) < vocab_size


def test_build_dummy_model_and_input_are_compatible(fake_transformers, fake_bitsandbytes) -> None:
    """build_dummy_model이 돌려준 config.vocab_size로 build_dummy_input을 바로 쓸 수 있다."""
    from preflight.canary.model import build_dummy_input, build_dummy_model

    _model, config, _quant_backend = build_dummy_model("some-org/some-model", device="cpu")
    input_ids = build_dummy_input(
        batch_size=1, seq_len=4, vocab_size=config.vocab_size, device="cpu"
    )

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

    # **모델 생성만 확인하면 안 된다.** #134는 여기까지 통과하고 forward에서 죽었다 —
    # RoPE의 inv_freq 버퍼가 meta로 남아 있었는데 이 테스트는 그걸 못 봤다.
    assert not [name for name, buf in model.named_buffers() if buf.device.type == "meta"]

    from preflight.canary.model import build_dummy_input

    dummy = build_dummy_input(1, 8, _config.vocab_size, "cuda")
    model(dummy)  # 죽지 않으면 성공 — 값은 보지 않는다(가중치가 랜덤이다)

    linear4bit_count = sum(1 for m in model.modules() if isinstance(m, bnb.nn.Linear4bit))
    assert linear4bit_count > 0, "Linear4bit 레이어가 하나도 없다 — 양자화가 실제로 안 걸린 것"


# ── config 조회 실패 메시지 (#62) ─────────────────────────────────────────────
#
# HF Hub는 없는 저장소에도 404가 아니라 401을 돌려준다. 그 401이 3단 체인으로
# 올라오면 error_log가 57줄이 되고 화면에는 "private repository … token"만 남아,
# 모델명을 잘못 친 사용자가 토큰을 찾으러 간다. 아는 원인은 여기서 확정한다.


def _config_error(monkeypatch, exc: Exception):
    """AutoConfig.from_pretrained이 주어진 예외로 실패하는 transformers를 끼운다."""

    class _RaisingAutoConfig:
        @staticmethod
        def from_pretrained(model_name):
            raise exc

    monkeypatch.setitem(
        sys.modules, "transformers", types.SimpleNamespace(AutoConfig=_RaisingAutoConfig)
    )


def _make_error(name: str, message: str, base=OSError) -> Exception:
    """huggingface_hub을 설치하지 않고 그 예외 타입 이름만 흉내 낸다."""
    return type(name, (base,), {})(message)


def test_load_config_repository_not_found_says_model_not_found(monkeypatch) -> None:
    from preflight.canary.model import ModelConfigError, _load_config

    _config_error(
        monkeypatch,
        _make_error("RepositoryNotFoundError", "401 Client Error. (Request ID: Root=1-68a)"),
    )

    with pytest.raises(ModelConfigError) as exc_info:
        _load_config("this-org-does-not-exist/definitely-not-a-model")

    message = str(exc_info.value)
    assert "모델을 찾을 수 없음" in message
    assert "this-org-does-not-exist/definitely-not-a-model" in message
    assert "401" not in message and "token" not in message
    # 체인을 끊어야 traceback.format_exc()가 3단 트레이스백을 다시 찍지 않는다.
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__ is True


def test_load_config_final_oserror_text_also_matches(monkeypatch) -> None:
    """hf_hub 예외 타입이 아니라 transformers의 최종 OSError로 올라오는 경우도 잡는다."""
    from preflight.canary.model import ModelConfigError, _load_config

    _config_error(
        monkeypatch,
        OSError(
            "this-org/nope is not a local folder and is not a valid model identifier "
            "listed on 'https://huggingface.co/models'\nIf this is a private repository, "
            "make sure to pass a token"
        ),
    )

    with pytest.raises(ModelConfigError) as exc_info:
        _load_config("this-org/nope")

    assert "모델을 찾을 수 없음" in str(exc_info.value)


def test_load_config_gated_repo_keeps_token_guidance(monkeypatch) -> None:
    """접근 제한 모델은 **진짜로** 토큰이 필요하다 — 그 안내는 여기서만 나온다."""
    from preflight.canary.model import ModelConfigError, _load_config

    _config_error(monkeypatch, _make_error("GatedRepoError", "403 Client Error"))

    with pytest.raises(ModelConfigError) as exc_info:
        _load_config("meta-llama/Llama-3.1-8B")

    message = str(exc_info.value)
    assert "접근이 제한된" in message
    assert "권한" in message


def test_load_config_offline_cache_miss_says_network(monkeypatch) -> None:
    """오프라인 + 캐시 미스는 모델명 문제가 아니라 네트워크 문제로 안내한다(#62)."""
    from preflight.canary.model import ModelConfigError, _load_config

    _config_error(
        monkeypatch,
        _make_error(
            "LocalEntryNotFoundError",
            "Cannot reach huggingface.co: Check your internet connection or see how to "
            "run the library in offline mode.",
        ),
    )

    with pytest.raises(ModelConfigError) as exc_info:
        _load_config("meta-llama/Llama-3.1-8B")

    message = str(exc_info.value)
    assert "네트워크" in message
    assert "캐시" in message


def test_load_config_unknown_failure_is_reraised_untouched(monkeypatch) -> None:
    """모르는 실패를 오타로 단정하지 않는다 — 그게 이 이슈의 버그였다(#62)."""
    from preflight.canary.model import ModelConfigError, _load_config

    original = ValueError("Unrecognized model in some-org/some-model")
    _config_error(monkeypatch, original)

    with pytest.raises(ValueError) as exc_info:
        _load_config("some-org/some-model")

    assert exc_info.value is original
    assert not isinstance(exc_info.value, ModelConfigError)


def test_build_dummy_model_surfaces_config_error(fake_bitsandbytes, monkeypatch) -> None:
    """build_dummy_model이 config 실패를 그대로 올린다 — 모델 구성 단계로 넘어가지 않는다."""
    from preflight.canary.model import ModelConfigError, build_dummy_model

    _config_error(monkeypatch, _make_error("RepositoryNotFoundError", "401 Client Error"))

    with pytest.raises(ModelConfigError):
        build_dummy_model("this-org-does-not-exist/definitely-not-a-model", device="cpu")


# --- meta 버퍼 실체화 (#134) ---


def test_no_meta_buffers_remain_after_materialization(fake_transformers) -> None:
    """실체화 후 meta로 남은 **버퍼**가 없어야 한다 (#134).

    `with torch.device("meta")`는 그 블록에서 만들어지는 **모든** 텐서를 meta로
    보낸다 — 파라미터와 버퍼를 구분하지 않는다. 그런데 실체화 루프가 파라미터만
    훑고 있어서, RoPE 계열(Llama·Mistral·Qwen)의 `inv_freq` 버퍼가 껍데기로 남아
    `--model`이 통째로 실패했다.

    이 테스트는 **GPU도 실물 라이브러리도 없이** 돈다 — 같은 경로를 검증하던
    `@pytest.mark.network` 테스트는 CI에서 아예 실행되지 않아 이 버그를 못 잡았다.
    """
    from preflight.canary.model import build_dummy_model

    model, _config, _backend = build_dummy_model("dummy/model", device="cpu")

    left_on_meta = [name for name, buf in model.named_buffers() if buf.device.type == "meta"]

    assert left_on_meta == [], left_on_meta


def test_materialized_buffer_keeps_shape_dtype_and_persistence(fake_transformers) -> None:
    """실체화한 버퍼가 원래 모양·자료형·persistent 여부를 유지한다 (#134).

    값은 0으로 채운다 — VRAM은 값이 아니라 모양·자료형으로 정해지므로 측정에 영향이
    없다. 반면 persistent 여부가 바뀌면 `state_dict` 구성이 달라져 의미가 있다.
    """
    from preflight.canary.model import build_dummy_model

    model, _config, _backend = build_dummy_model("dummy/model", device="cpu")

    buf = model.get_buffer("inv_freq")

    assert buf.shape == (8,)
    assert buf.device.type == "cpu"
    # persistent=False로 등록했으므로 state_dict에 나오면 안 된다.
    assert "inv_freq" not in model.state_dict()


def test_forward_runs_after_materialization(fake_transformers) -> None:
    """실체화한 모델로 forward가 실제로 돈다 (#134).

    **이 단정이 이 이슈의 핵심이다.** 기존 테스트는 `Linear4bit` 개수만 세고 forward를
    돌리지 않아, 모델은 만들어지는데 돌리면 죽는 상태를 통과시켰다.
    """
    from preflight.canary.model import build_dummy_model

    model, _config, _backend = build_dummy_model("dummy/model", device="cpu")

    # dtype을 모델에 맞춘다 — from_config(dtype=float16)로 만들어진다.
    weight = next(model.parameters())
    model(torch.randn(1, 8, dtype=weight.dtype))
