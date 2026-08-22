"""MODULE-03 ReVerifier — 수정 직후 canary를 동일 조건으로 재실행해 PASS 전환 여부를 확정한다.

docs/architecture.md §5 MODULE-03 참고. --yes가 있을 때만 호출된다
(docs/contracts/canary-api.md §5.4 전체 흐름 5·6번).
"""

from __future__ import annotations

from preflight.canary.engine import run_canary_check
from preflight.canary.judge import judge_result
from preflight.gpu import query_gpu_state


def reverify(model_name: str | None, batch_size: int, seq_len: int) -> dict:
    """수정 명령어 실행 직후 canary를 동일 조건으로 재실행해 판정 결과를 반환한다.

    **"동일 조건"에는 GPU 상태 조회도 포함된다**(#68). 예전에는
    `query_gpu_state()`를 부르지 않아 `env.gpu_free_mb`가 비었는데,
    `judge_result`는 그 값이 없으면 `memory_delta_high` WARN을 조용히 건너뛴다
    (canary-api.md에 명시된 동작). 그래서 VRAM이 빠듯해 WARN이 떴던 환경도
    재확인에서는 그 WARN이 **구조적으로 다시 나올 수 없어** 늘 PASS로 뒤집혔다.

    조회 시점은 1차 실행 때가 아니라 지금이다 — 알고 싶은 게 "수정 이후 시점의
    여유 VRAM"이라 값이 1차와 달라지는 것이 맞는 동작이다.
    """
    state = query_gpu_state()

    raw = run_canary_check(
        model_name=model_name,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    if state:
        raw["env"] = {
            **(raw.get("env") or {}),
            "gpu_free_mb": state["free_mb"],
            "gpu_total_mb": state["total_mb"],
        }

    return judge_result(raw)
