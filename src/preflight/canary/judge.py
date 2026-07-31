"""판정 로직(PASS/WARN/FAIL). docs/contracts/canary-api.md의 judge_result 계약을 구현한다.

두 임계값은 매직넘버로 우선 채택한 것이며, 실측 데이터가 쌓이면 조정될 수 있다.
"""

from __future__ import annotations

MEMORY_DELTA_WARN_PCT = 15
CPU_MULTIPLIER_WARN_THRESHOLD = 2


def judge_result(raw: dict) -> dict:
    raise NotImplementedError
