"""rich 기반 CLI 리포트 출력 및 --json 직렬화. 출력 예시는 docs/contracts/cli.md 참고."""

from __future__ import annotations


def render_report(results: list[dict], json_output: bool = False) -> None:
    raise NotImplementedError
