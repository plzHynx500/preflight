"""MODULE-01 CanaryEngine. docs/contracts/canary-api.md의 run_canary_check 계약을 구현한다."""

from __future__ import annotations


def run_canary_check(model, batch_size: int, seq_len: int) -> dict:
    """canary/worker.py를 subprocess로 격리 실행하고 원시 측정값을 반환한다.

    반환 스키마 (status/device/memory_delta_mb/elapsed_ms/cpu_multiplier/error_log)는
    docs/contracts/canary-api.md 참고. 절대 예외를 던지거나 프로세스를 죽이지 않는다.
    """
    raise NotImplementedError
