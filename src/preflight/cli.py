"""Typer 진입점. 명령어 계약은 docs/contracts/cli.md 참고."""

from __future__ import annotations

import time
from typing import Optional

import typer

from preflight.canary.engine import run_canary_check
from preflight.canary.judge import judge_result
from preflight.fix.executor import FixExecutionError, apply_fix, suggest_fix
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


def _aggregate_verdict(results: list[dict]) -> str:
    """결과 여러 개를 하나의 최종 판정으로 접는다 (docs/contracts/cli.md "결과 집계").

    판정 필드가 없는 항목(fail-fast로 생략된 체크)은 판정된 적이 없으므로 뺀다.
    """
    verdicts = [r["verdict"] for r in results if "verdict" in r]
    return "FAIL" if "FAIL" in verdicts else "WARN" if "WARN" in verdicts else "PASS"


def _select_fix_target(results: list[dict]) -> dict | None:
    """FIX를 붙일 항목 하나를 고른다 — **FAIL을 WARN보다 먼저 본다**(#69).

    예전에는 결과 배열에서 처음 만난 non-PASS 항목에 붙였다. 기본 체크가 WARN,
    모델 체크가 FAIL이면 순서상 WARN이 먼저라, 화면에는 OOM FAIL을 띄워놓고
    "CPU 대비 연산 속도 2배 미만" 안내를 하는 상황이 났다 — 정작 FAIL에 대한
    안내는 어디에도 없었다.

    FIX를 하나만 보여주는 구조는 MVP 그대로 유지한다(#69 범위 제외). FAIL이
    여럿이면 그중 첫 번째다.
    """
    for wanted in ("FAIL", "WARN"):
        for result in results:
            if result.get("verdict") != wanted:
                continue
            fix = suggest_fix(result)
            if fix:
                result["fix"] = fix
                return result
    return None


def _describe_fix_failure(error: FixExecutionError, command: str) -> str:
    """수정 명령 실패를 진단 결과와 같은 톤의 한 줄로 바꾼다 (#53).

    pip의 stdout/stderr는 싣지 않는다 — `FixExecutionError` 메시지에는 그게
    통째로 들어 있어 그대로 흘리면 화면이 길게 쏟아진다. 출력 자체의 요약·가공은
    #53 범위 밖이라, 대신 "직접 실행해 보라"고 안내한다.
    """
    if error.returncode is None:
        detail = "명령을 실행조차 하지 못했다"
    else:
        detail = f"종료 코드 {error.returncode}"
    return (
        f"자동 수정 실패: {command} ({detail}) — 재확인을 건너뛰었다."
        " 위 명령을 직접 실행해 오류를 확인한 뒤 preflight check로 다시 확인하세요."
    )


@app.command()
def check(
    # `from __future__ import annotations`가 있어도 여기서는 `X | None`을 쓸 수 없다 —
    # Typer가 CLI 파서를 만들려고 런타임에 `typing.get_type_hints()`로 이 문자열
    # 어노테이션을 다시 평가하기 때문이다. 그 순간 Python 3.9에서 `str | None`이
    # 실제로 계산되어 TypeError가 난다(#42). requires-python이 ">=3.9"인 한
    # 런타임에 읽히는 어노테이션은 Optional[...] 표기를 유지해야 한다.
    model: Optional[str] = typer.Option(None, "--model", help="HuggingFace 모델명 또는 config"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size"),
    seq_len: Optional[int] = typer.Option(None, "--seq-len"),
    yes: bool = typer.Option(False, "--yes", help="제시된 수정 명령어를 실행하고 재확인까지 수행"),
    json_output: bool = typer.Option(False, "--json", help="JSON 형식으로 결과 출력"),
) -> None:
    """docs/contracts/canary-api.md §5.4 전체 흐름을 따른다."""
    start_time = time.perf_counter()

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

    verdict = _aggregate_verdict(results)

    fix_target = _select_fix_target(results) if verdict != "PASS" else None
    fix = fix_target.get("fix") if fix_target else None

    notices: list[str] = []
    if yes and fix:
        command = fix.get("fix_command")
        if not fix.get("fix_argv"):
            # `suggest_fix`는 fix_command가 없어도 dict를 돌려주므로 `if yes and fix:`
            # 만으로는 항상 참이었다. 그래서 실행할 명령이 하나도 없는 환경에서도
            # canary를 통째로 한 번 더 돌리고(실측 11초 → 17초), 사용자에게는 뭘
            # 했는지 한 마디도 하지 않았다(#57).
            notices.append(
                "--yes: 자동으로 실행할 수정 명령이 없어 아무것도 실행하지 않았다"
                " — 위 안내를 보고 직접 조치해야 한다."
            )
        else:
            try:
                apply_fix(fix)
            except FixExecutionError as error:
                # 트레이스백을 그대로 흘리면, 에러를 읽기 쉽게 만들어주는 것이 목적인
                # 도구가 정확히 그 지점에서 스스로 무너진다(#53). 재확인은 건너뛰고
                # 종료 코드는 1차 판정 그대로 — 재확인을 한 적이 없기 때문이다.
                notices.append(_describe_fix_failure(error, str(command)))
            else:
                notices.append(f"자동 수정 실행: {command}")
                # 재확인은 **fix의 근거가 된 그 체크**를 같은 조건으로 다시 돌린다.
                # 예전에는 --model이 주어졌으면 언제나 모델 canary를 돌렸는데,
                # 기본 체크가 FAIL이면 모델 체크는 fail-fast로 한 번도 실행된 적이
                # 없는 상태다 — 고쳤는지 확인해야 할 기본 체크는 재실행되지 않고,
                # 대신 처음 보는 체크가 돌아갔다(#68).
                reverified = reverify(
                    model_name=fix_target.get("model_name"),
                    batch_size=fix_target.get("batch_size", 1),
                    seq_len=fix_target.get("seq_len", 8),
                )
                # 1차 결과를 재확인 결과로 갈아끼운다 — 그러지 않으면 수정이
                # 성공해도 화면에는 1차의 ✖ 줄이 그대로 남고 종료 코드만 0이 된다
                # (#57). CLI가 얹은 값(model_name·batch_size·seq_len)과 실행한
                # fix는 유지하고, 자식이 잰 값은 통째로 새 측정값을 쓴다.
                carried = {
                    key: fix_target[key]
                    for key in ("model_name", "batch_size", "seq_len", "fix")
                    if key in fix_target
                }
                # `list.index()`는 dict를 값으로 비교하므로 내용이 같은 다른 항목을
                # 집을 수 있다 — 갈아끼울 자리는 동일성으로 찾는다.
                index = next(i for i, r in enumerate(results) if r is fix_target)
                results[index] = {**reverified, **carried, "reverified": True}
                # 재확인 1건으로 전체 verdict를 대체하지 않는다 — 그러면 재확인하지
                # 않은 나머지 결과(예: 기본 체크의 WARN)가 종료 코드에서 사라진다(#68).
                verdict = _aggregate_verdict(results)

    elapsed_seconds = time.perf_counter() - start_time
    render_report(
        results,
        json_output=json_output,
        elapsed_seconds=elapsed_seconds,
        notices=notices,
    )

    exit_code = get_exit_code(verdict)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    app()
