"""Typer 진입점. 명령어 계약은 docs/contracts/cli.md 참고."""

from __future__ import annotations

import typer

from preflight.canary.engine import run_canary_check
from preflight.canary.judge import judge_result
from preflight.fix.executor import apply_fix, suggest_fix
from preflight.report import render_report
from preflight.reverify import reverify

app = typer.Typer(help="Preflight — 파인튜닝 환경 진단 CLI")


@app.command()
def check(
    model: str | None = typer.Option(None, "--model", help="HuggingFace 모델명 또는 config"),
    batch_size: int | None = typer.Option(None, "--batch-size"),
    seq_len: int | None = typer.Option(None, "--seq-len"),
    yes: bool = typer.Option(False, "--yes", help="제시된 수정 명령어를 실행하고 재확인까지 수행"),
    json_output: bool = typer.Option(False, "--json", help="JSON 형식으로 결과 출력"),
) -> None:
    """docs/contracts/canary-api.md §5.4 전체 흐름을 따른다."""
    raw = run_canary_check(
        model=model,
        batch_size=batch_size or 1,
        seq_len=seq_len or 8,
    )
    check_result = judge_result(raw)
    verdict = check_result.get("verdict", "PASS")

    fix = None
    if verdict != "PASS":
        fix = suggest_fix(check_result)
        if fix:
            check_result["fix"] = fix

    if yes and fix:
        apply_fix(fix)
        reverified = reverify(
            model=model,
            batch_size=batch_size or 1,
            seq_len=seq_len or 8,
        )
        check_result["reverified"] = reverified
        verdict = reverified.get("verdict", "PASS")

    render_report([check_result], json_output=json_output)

    if verdict != "PASS":
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
