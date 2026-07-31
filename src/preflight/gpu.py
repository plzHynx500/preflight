"""pynvml 기반 GPU/드라이버 상태 조회. docs/architecture.md §2 기술 스택 참고."""

from __future__ import annotations


def query_gpu_state() -> dict:
    raise NotImplementedError
