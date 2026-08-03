"""build_dummy_model/build_dummy_input 테스트.

이 환경에는 transformers가 없어(pyproject.toml이 의도적으로 하드 의존성에
넣지 않음 — docs/architecture.md §2 참고) `sys.modules`에 가짜 transformers를
심어 로직만 검증한다. 실제 transformers/모델 config를 검증하는 통합 테스트는
CI에 transformers가 설치되면 별도로 추가한다.
"""

from __future__ import annotations

import sys
import types

import pytest


class _FakeConfig:
    def __init__(self, vocab_size: int = 32000) -> None:
        self.vocab_size = vocab_size


class _FakeModel:
    def __init__(self, config: _FakeConfig) -> None:
        self.config = config


@pytest.fixture
def fake_transformers(monkeypatch):
    """AutoConfig.from_pretrained·AutoModelForCausalLM.from_config을 흉내낸다."""
    calls = {}

    class _FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_name):
            calls["config_model_name"] = model_name
            return _FakeConfig()

    class _FakeAutoModelForCausalLM:
        @staticmethod
        def from_config(config):
            calls["from_config_arg"] = config
            return _FakeModel(config)

    fake_module = types.SimpleNamespace(
        AutoConfig=_FakeAutoConfig,
        AutoModelForCausalLM=_FakeAutoModelForCausalLM,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_module)
    return calls


def test_build_dummy_model_uses_config_only_no_weights(fake_transformers) -> None:
    from preflight.canary.model import build_dummy_model

    model, config = build_dummy_model("some-org/some-model")

    assert fake_transformers["config_model_name"] == "some-org/some-model"
    # from_config()에 넘긴 것이 from_pretrained()가 조회해온 그 config여야 한다
    # (가중치 다운로드 경로인 from_pretrained(model)이 아니다).
    assert fake_transformers["from_config_arg"] is config
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


def test_build_dummy_model_and_input_are_compatible(fake_transformers) -> None:
    """build_dummy_model이 돌려준 config.vocab_size로 build_dummy_input을 바로 쓸 수 있다."""
    from preflight.canary.model import build_dummy_input, build_dummy_model

    _model, config = build_dummy_model("some-org/some-model")
    input_ids = build_dummy_input(batch_size=1, seq_len=4, vocab_size=config.vocab_size)

    assert input_ids.shape == (1, 4)
