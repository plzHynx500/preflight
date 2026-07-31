"""Typer 진입점. 명령어 계약은 docs/contracts/cli.md 참고."""

from __future__ import annotations

import typer

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
    raise NotImplementedError


if __name__ == "__main__":
    app()
