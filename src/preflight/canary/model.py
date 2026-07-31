"""MODULE-01 CanaryEngine 1~2단계 — config만 조회해 랜덤 초기화 모델·더미 입력을 구성한다.

가중치는 다운로드하지 않는다 (docs/architecture.md §5 MODULE-01 참고).
"""

from __future__ import annotations


def build_dummy_model(model_name: str | None):
    """AutoConfig.from_pretrained() + from_config()로 랜덤 초기화 모델을 구성한다."""
    raise NotImplementedError


def build_dummy_input(batch_size: int, seq_len: int):
    raise NotImplementedError
