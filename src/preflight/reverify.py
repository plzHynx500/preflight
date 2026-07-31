"""MODULE-03 ReVerifier — 수정 직후 canary를 동일 조건으로 재실행해 PASS 전환 여부를 확정한다.

docs/architecture.md §5 MODULE-03 참고. --yes가 있을 때만 호출된다
(docs/contracts/canary-api.md §5.4 전체 흐름 5·6번).
"""

from __future__ import annotations

from preflight.canary.engine import run_canary_check


def reverify(model, batch_size: int, seq_len: int) -> dict:
    raise NotImplementedError
