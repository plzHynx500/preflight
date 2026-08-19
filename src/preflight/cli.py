"""Typer 진입점. 명령어 계약은 docs/contracts/cli.md 참고."""

from __future__ import annotations

import time

import typer

from preflight.canary.engine import run_canary_check
from preflight.canary.judge import judge_result
from preflight.fix.executor import apply_fix, suggest_fix
from preflight.gpu import query_gpu_state
from preflight.report import render_report
from preflight.reverify import reverify

app = typer.Typer(help="Preflight — 파인튜닝 환경 진단 CLI")


@app.callback()
def callback() -> None:
    """Preflight CLI."""


def get_exit_code(verdict: str) -> int:
    """판정 결과(PASS/WARN/FAIL)에 따른 CLI 종료 코드 반환.

    - 0: 모든 항목이 PASS인 경우
    - 1: FAIL이 하나라도 포함된 경우
    - 2: FAIL 없이 WARN만 포함된 경우
    """
    v = verdict.upper() if verdict else "PASS"
    if v == "FAIL":
        return 1
    if v == "WARN":
        return 2
    return 0


@app.command()
def check(
    model: str | None = typer.Option(None, "--model", help="HuggingFace 모델명 또는 config"),
    batch_size: int | None = typer.Option(None, "--batch-size"),
    seq_len: int | None = typer.Option(None, "--seq-len"),
    yes: bool = typer.Option(False, "--yes", help="제시된 수정 명령어를 실행하고 재확인까지 수행"),
    json_output: bool = typer.Option(False, "--json", help="JSON 형식으로 결과 출력"),
) -> None:
    """docs/contracts/canary-api.md §5.4 전체 흐름을 따른다."""
    start_time = time.monotonic()

    state = query_gpu_state()

    raw_basic = run_canary_check(
        model_name=None,
        batch_size=1,
        seq_len=8,
    )
    if state:
        raw_basic["env"] = {
            **(raw_basic.get("env") or {}),
            "gpu_free_mb": state["free_mb"],
            "gpu_total_mb": state["total_mb"],
        }

    basic_res = judge_result(raw_basic)
    results = [basic_res]

    if model:
        meta = {"model_name": model, "batch_size": batch_size or 1, "seq_len": seq_len or 8}
        if basic_res.get("verdict") == "FAIL":
            results.append({**meta, "skipped": "환경 체크 실패"})
        else:
            raw_model = run_canary_check(
                model_name=model,
                batch_size=batch_size or 1,
                seq_len=seq_len or 8,
            )
            if state:
                raw_model["env"] = {
                    **(raw_model.get("env") or {}),
                    "gpu_free_mb": state["free_mb"],
                    "gpu_total_mb": state["total_mb"],
                }
            # 이미 잰 값을 그대로 쓴다 — 재측정하면 canary 자신의 점유만큼 깎여 오염된다
            model_res = judge_result(raw_model)
            results.append({**model_res, **meta})

    verdicts = [r["verdict"] for r in results if "verdict" in r]
    verdict = "FAIL" if "FAIL" in verdicts else "WARN" if "WARN" in verdicts else "PASS"

    fix = None
    if verdict != "PASS":
        # 실패한 결과 중 첫 번째 항목에 대해 fix를 제안한다
        for r in results:
            if r.get("verdict") != "PASS":
                fix = suggest_fix(r)
                if fix:
                    r["fix"] = fix
                    break

    if yes and fix:
        apply_fix(fix)
        reverified = reverify(
            model=model,
            batch_size=batch_size or 1,
            seq_len=seq_len or 8,
        )
        # reverified 결과도 results에 얹어줄 수 있지만, 지금은 마지막 reverified 객체 자체를 리포트에 넣는 것으로 유지
        for r in results:
            if "fix" in r:
                r["reverified"] = reverified
                break
        verdict = reverified.get("verdict", "PASS")

    elapsed_seconds = time.monotonic() - start_time
    render_report(results, json_output=json_output, elapsed_seconds=elapsed_seconds)

    exit_code = get_exit_code(verdict)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    app()
