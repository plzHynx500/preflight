"""build_dummy_model/build_dummy_input 테스트.

이 환경에는 transformers/peft가 없어(pyproject.toml이 의도적으로 하드 의존성에
넣지 않음 — docs/architecture.md §2 참고) `sys.modules`에 가짜 모듈을 심어
로직만 검증한다. 실제 transformers/peft/bitsandbytes를 검증하는 통합 테스트는
CI에 그 패키지들이 설치되면 별도로 추가한다.
"""

from __future__ import annotations

import sys
import types

import pytest


class _FakeConfig:
    def __init__(self, vocab_size: int = 32000) -> None:
        self.vocab_size = vocab_size


class _FakeModel:
    def __init__(self, config: _FakeConfig, quantized: bool = False) -> None:
        self.config = config
        self.quantized = quantized


class _FakePeftModel:
    def __init__(self, base_model, lora_config) -> None:
        self.base_model = base_model
        self.lora_config = lora_config
        self.config = base_model.config


@pytest.fixture
def fake_transformers(monkeypatch):
    """AutoConfig.from_pretrained·AutoModelForCausalLM.from_config을 흉내낸다.

    from_config은 quantization_config가 있으면 양자화된 모델을,
    없으면(fp32 폴백 경로) 평범한 모델을 돌려준다.
    """
    calls = {}

    class _FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_name):
            calls["config_model_name"] = model_name
            return _FakeConfig()

    class _FakeAutoModelForCausalLM:
        @staticmethod
        def from_config(config, quantization_config=None):
            calls["from_config_arg"] = config
            calls["quantization_config"] = quantization_config
            return _FakeModel(config, quantized=quantization_config is not None)

    class _FakeBitsAndBytesConfig:
        def __init__(self, **kwargs) -> None:
            calls["bnb_config_kwargs"] = kwargs

    fake_module = types.SimpleNamespace(
        AutoConfig=_FakeAutoConfig,
        AutoModelForCausalLM=_FakeAutoModelForCausalLM,
        BitsAndBytesConfig=_FakeBitsAndBytesConfig,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_module)
    return calls


@pytest.fixture
def fake_peft(monkeypatch):
    """LoraConfig·get_peft_model을 흉내낸다."""
    calls = {}

    class _FakeLoraConfig:
        def __init__(self, **kwargs) -> None:
            calls["lora_config_kwargs"] = kwargs

    def _fake_get_peft_model(model, lora_config):
        calls["get_peft_model_args"] = (model, lora_config)
        return _FakePeftModel(model, lora_config)

    fake_module = types.SimpleNamespace(
        LoraConfig=_FakeLoraConfig,
        get_peft_model=_fake_get_peft_model,
    )
    monkeypatch.setitem(sys.modules, "peft", fake_module)
    return calls


def _remove_real_peft(monkeypatch) -> None:
    """peft가 실제로 설치돼 있어도(이 환경엔 없지만) 폴백 경로 테스트에선 없는 것처럼 만든다."""
    monkeypatch.delitem(sys.modules, "peft", raising=False)
    monkeypatch.setattr(
        "builtins.__import__",
        _make_blocking_import({"peft"}),
    )


def _make_blocking_import(blocked_names: set[str]):
    real_import = __import__

    def _blocking_import(name, *args, **kwargs):
        if name in blocked_names or name.split(".")[0] in blocked_names:
            raise ModuleNotFoundError(f"No module named {name!r} (blocked for test)")
        return real_import(name, *args, **kwargs)

    return _blocking_import


def test_build_dummy_model_uses_config_only_no_weights(fake_transformers, fake_peft) -> None:
    from preflight.canary.model import build_dummy_model

    model, config, quant_backend = build_dummy_model("some-org/some-model")

    assert fake_transformers["config_model_name"] == "some-org/some-model"
    # from_config()에 넘긴 것이 from_pretrained()가 조회해온 그 config여야 한다
    # (가중치 다운로드 경로인 from_pretrained(model)이 아니다).
    assert fake_transformers["from_config_arg"] is config
    assert model.config is config
    assert quant_backend == "bnb-4bit"


def test_build_dummy_model_applies_4bit_and_lora_when_available(
    fake_transformers, fake_peft
) -> None:
    """QLoRA는 옵션이 아니라 고정 가정이다 — 항상 4bit+LoRA를 시도한다(#12 리뷰, 상영님 지적)."""
    from preflight.canary.model import MINIMAL_ADAPTER_RANK, build_dummy_model

    model, _config, quant_backend = build_dummy_model("some-org/some-model")

    assert quant_backend == "bnb-4bit"
    assert fake_transformers["quantization_config"] is not None
    assert fake_transformers["bnb_config_kwargs"]["load_in_4bit"] is True
    lora_kwargs = fake_peft["lora_config_kwargs"]
    assert lora_kwargs["r"] == MINIMAL_ADAPTER_RANK
    assert lora_kwargs["task_type"] == "CAUSAL_LM"
    # LoRA 어댑터가 양자화된 모델 위에 씌워져야 한다 — 순서가 바뀌면 안 된다.
    peft_model_arg, _lora_config_arg = fake_peft["get_peft_model_args"]
    assert peft_model_arg.quantized is True
    assert model.base_model is peft_model_arg


def test_build_dummy_model_falls_back_to_fp32_when_peft_missing(
    fake_transformers, monkeypatch
) -> None:
    """peft가 없는 환경에서도 죽지 않고 fp32 전체 모델로 폴백한다."""
    _remove_real_peft(monkeypatch)

    from preflight.canary.model import build_dummy_model

    model, config, quant_backend = build_dummy_model("some-org/some-model")

    assert quant_backend == "nn-linear-fallback"
    # 폴백 경로는 quantization_config 없이 from_config를 호출한다.
    assert fake_transformers["quantization_config"] is None
    assert model.config is config


def test_build_dummy_input_uses_vocab_size_from_config() -> None:
    import torch

    from preflight.canary.model import build_dummy_input

    vocab_size = 128
    input_ids = build_dummy_input(batch_size=2, seq_len=6, vocab_size=vocab_size)

    assert input_ids.shape == (2, 6)
    assert input_ids.dtype in (torch.int64, torch.int32)
    assert int(input_ids.min()) >= 0
    assert int(input_ids.max()) < vocab_size


def test_build_dummy_model_and_input_are_compatible(fake_transformers, fake_peft) -> None:
    """build_dummy_model이 돌려준 config.vocab_size로 build_dummy_input을 바로 쓸 수 있다."""
    from preflight.canary.model import build_dummy_input, build_dummy_model

    _model, config, _quant_backend = build_dummy_model("some-org/some-model")
    input_ids = build_dummy_input(batch_size=1, seq_len=4, vocab_size=config.vocab_size)

    assert input_ids.shape == (1, 4)
