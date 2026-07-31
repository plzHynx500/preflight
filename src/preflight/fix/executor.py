"""MODULE-02 FixExecutor. docs/contracts/canary-api.md의 suggest_fix 계약을 구현한다."""

from __future__ import annotations


def suggest_fix(check_result: dict) -> dict | None:
    """실행하지 않는다 — 명령어 텍스트만 반환. verdict가 PASS면 None."""
    raise NotImplementedError


def apply_fix(fix: dict) -> None:
    """--yes 지정 시에만 호출된다."""
    raise NotImplementedError
