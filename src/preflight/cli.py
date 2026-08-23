"""Typer 진입점. 명령어 계약은 docs/contracts/cli.md 참고."""

from __future__ import annotations

import sys
import time
from typing import Optional

import typer

from preflight import __version__
from preflight.canary.engine import run_canary_check
from preflight.canary.judge import judge_result
from preflight.fix.executor import FixExecutionError, apply_fix, suggest_fix
from preflight.gpu import query_gpu_state
from preflight.report import render_report
from preflight.reverify import reverify

app = typer.Typer(help="Preflight — 파인튜닝 환경 진단 CLI")


def ensure_utf8_streams() -> None:
    """stdout/stderr을 UTF-8로 강제한다.

    Windows에서 리다이렉트(`> file`, `| other`)된 스트림은 콘솔이 아니라 시스템
    로케일(cp949 등) 인코딩을 따른다 — 대화형 콘솔은 PEP 528 덕에 UTF-8이라 개발
    중에는 드러나지 않는다. 리포트의 `✔/✖/⚠/…/—` 같은 기호가 그 로케일에 없으면
    `UnicodeEncodeError`로 죽어 `preflight check > result.txt`,
    `preflight check --json > result.json` 같은 CI 연동 시나리오가 통째로
    실패한다(#54). `errors="replace"`로 두어 그래도 못 옮기는 극단적 환경에서는
    죽는 대신 `?`로 대체한다.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


# 모듈 임포트 시점에 실행한다 — @app.callback()에 두면 Click이 최상위(그룹)
# --help를 콜백 실행 전에 처리하고 종료해, `preflight --help`를 리다이렉트할 때
# 콜백을 한 번도 안 거치고 rich 도움말 상자의 박스 문자에서 그대로 죽는다(#89).
# 서브커맨드(`check --help`)는 그룹 콜백이 먼저 돌아 우연히 살아났던 것뿐이다.
# 부작용이 스트림 인코딩 재설정뿐이고 `reconfigure` 존재 여부도 가드하므로
# 임포트 시점 실행이 안전하다.
ensure_utf8_streams()


def _version_callback(show_version: bool) -> None:
    if show_version:
        typer.echo(f"preflight {__version__}")
        raise typer.Exit()


@app.callback()
def callback(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="버전 정보를 출력하고 종료",
    ),
) -> None:
    """Preflight CLI."""


def _progress(message: str) -> None:
    """진행 상황을 stderr에 한 줄 찍는다 (#63).

    기본 체크는 첫 줄까지 10초 안팎, `--yes`는 pip 재설치+canary 재실행으로 1분
    넘게 아무 출력이 없어 멈춘 것처럼 보인다. stdout은 `--json` 계약(cli.md)상
    결과만 담아야 하므로 stderr를 쓴다.

    직후 `flush()`가 필요하다 — 안 하면 뒤이어 돌 몇 초~몇십 초짜리 canary/pip
    실행 동안 이 줄이 파이프 버퍼에 잠겨 있다가 나중에야 나온다. 화면에 먼저
    보여주는 것이 이 기능의 목적이라 지연되면 의미가 없다(리다이렉트되면
    표준 스트림이 줄 단위 대신 블록 단위로 버퍼링되는 게 원인).
    """
    print(message, file=sys.stderr)
    sys.stderr.flush()


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


#: `--model` 의 VRAM 실측값이 무엇을 기준으로 잰 값인지 밝히는 안내(#118).
#:
#: canary 는 `AutoModelForCausalLM` 으로 직접 로드해 **vanilla(eager) 경로**로
#: 실행한다. Unsloth 처럼 attention·RoPE·cross-entropy 커널을 몽키패치하는
#: 프레임워크를 쓰면 실사용량이 이 수치보다 작다 — 방향은 안전한 쪽이지만,
#: **우리가 부족하다고 말한 환경에서 실제로는 학습이 되는 false negative** 가
#: 남는다. SRS §3 의 1번 사용자 시나리오가 곧 unsloth 사용자다.
#:
#: 수치를 보정하거나 프레임워크를 감지하지 않는다 — 감지·분기 실행은 FR-14(3),
#: 프레임워크 독립성은 NFR-06 이다. 여기서는 한계만 밝힌다.
_VANILLA_PATH_NOTICE = (
    "참고: VRAM 실측값은 vanilla(HuggingFace 기본) 실행 경로 기준입니다. "
    "Unsloth 등 커널 최적화 프레임워크를 쓰면 실제 사용량은 이보다 작을 수 있습니다."
)


def _model_check_ran(results: list[dict]) -> bool:
    """모델 체크가 실제로 돌아 VRAM 수치가 화면에 있는가 (#118).

    `--model` 을 줬어도 기본 체크가 FAIL 이면 모델 체크는 생략된다. 그때는
    보여줄 VRAM 수치가 없으므로 **그 수치의 기준을 설명하는 안내도 내지 않는다**
    — 없는 숫자에 대한 주석은 잡음이다.
    """
    return any(result.get("model_name") and "skipped" not in result for result in results)


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


def _reject_empty_model(value: Optional[str]) -> Optional[str]:
    """`--model ""`을 파싱 단계에서 거부한다 (#126).

    빈 문자열은 파이썬에서 falsy라 `if model:`이 "값 없음(`None`)"과 구분하지
    못해, 뒤에서 모델 체크 자체가 조용히 사라졌다 — CI에서 `--model "$MODEL"`의
    `$MODEL`이 비었을 때 특히 위험하다(에러도 경고도 없이 검증이 하나 빠진 채
    통과). `--batch-size 0`(#59)과 같은 이유로 canary 실행 전에 거부한다.
    공백만 있는 문자열은 이 검증 대상이 아니다 — 이미 "주어진 값"으로 정상
    처리되어 모델 체크가 시도된다.
    """
    if value == "":
        raise typer.BadParameter(
            "빈 문자열은 허용되지 않는다 (모델명을 지정하거나 --model 자체를 생략하세요)"
        )
    return value


def _reject_size_args_without_model(
    model: Optional[str], batch_size: Optional[int], seq_len: Optional[int]
) -> None:
    """`--model` 없이 `--batch-size`·`--seq-len`을 주면 거부한다 (#150).

    이 둘은 모델 체크 전용이다(docs/contracts/cli.md) — 기본 체크는 판정 임계값의
    근거가 걸린 고정 크기(batch=1, seq=8)로만 돈다([ADR-0004]). 예전에는 `--model`
    없이 주면 조용히 무시됐다 — 사용자는 준 크기로 쟀다고 믿지만 실제로는 1×8을
    잰 결과를 본다(거짓 안심). Typer 옵션 콜백은 다른 옵션 값을 볼 수 없어(여기서는
    `--model` 값을 봐야 함) `_reject_empty_model`처럼 콜백으로 두지 못하고 명령
    본문 초입에서 검사한다.
    """
    if model:
        return
    given = [
        name
        for name, value in (("--batch-size", batch_size), ("--seq-len", seq_len))
        if value is not None
    ]
    if not given:
        return
    hint = "/".join(given)
    raise typer.BadParameter(
        f"{hint}은 --model 없이 쓸 수 없다 (모델 체크 전용) — --model과 함께 쓰거나 {hint}를 빼세요.",
        param_hint=given[0],
    )


@app.command()
def check(
    # `from __future__ import annotations`가 있어도 여기서는 `X | None`을 쓸 수 없다 —
    # Typer가 CLI 파서를 만들려고 런타임에 `typing.get_type_hints()`로 이 문자열
    # 어노테이션을 다시 평가하기 때문이다. 그 순간 Python 3.9에서 `str | None`이
    # 실제로 계산되어 TypeError가 난다(#42). requires-python이 ">=3.9"인 한
    # 런타임에 읽히는 어노테이션은 Optional[...] 표기를 유지해야 한다.
    model: Optional[str] = typer.Option(
        None, "--model", callback=_reject_empty_model, help="HuggingFace 모델명 또는 config"
    ),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size", min=1, help="모델 체크의 배치 크기 (기본 1, 1 이상 정수)"
    ),
    seq_len: Optional[int] = typer.Option(
        None, "--seq-len", min=1, help="모델 체크의 시퀀스 길이 (기본 8, 1 이상 정수)"
    ),
    yes: bool = typer.Option(False, "--yes", help="제시된 수정 명령어를 실행하고 재확인까지 수행"),
    json_output: bool = typer.Option(False, "--json", help="JSON 형식으로 결과 출력"),
) -> None:
    """GPU/CUDA 환경이 실제로 파인튜닝 가능한지 canary 연산으로 진단한다."""
    _reject_size_args_without_model(model, batch_size, seq_len)

    # 상세 흐름: docs/contracts/canary-api.md §5.4
    start_time = time.perf_counter()

    state = query_gpu_state()

    _progress("진단 중… (torch 불러오기 · canary 실행, 수 초~수십 초)")
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
            "gpu_driver_version": state["driver_version"],
            "gpu_name": state["name"],
        }

    basic_res = judge_result(raw_basic)
    results = [basic_res]

    if model:
        meta = {"model_name": model, "batch_size": batch_size or 1, "seq_len": seq_len or 8}
        if basic_res.get("verdict") == "FAIL":
            results.append({**meta, "skipped": "환경 체크 실패"})
        else:
            _progress("진단 중… (torch 불러오기 · canary 실행, 수 초~수십 초)")
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
                    "gpu_driver_version": state["driver_version"],
                    "gpu_name": state["name"],
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
            _progress(f"수정 명령 실행 중: {command}")
            try:
                apply_fix(fix)
            except FixExecutionError as error:
                # 트레이스백을 그대로 흘리면, 에러를 읽기 쉽게 만들어주는 것이 목적인
                # 도구가 정확히 그 지점에서 스스로 무너진다(#53). 재확인은 건너뛰고
                # 종료 코드는 1차 판정 그대로 — 재확인을 한 적이 없기 때문이다.
                notices.append(_describe_fix_failure(error, str(command)))
            else:
                notices.append(f"자동 수정 실행: {command}")
                _progress("재확인 중…")
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
                # (#57). CLI가 얹은 값(model_name·batch_size·seq_len)은 유지하고,
                # 자식이 잰 값은 통째로 새 측정값을 쓴다.
                #
                # `fix`도 여기서 일단 옮기지만 **그대로 남지 않는다** — 아래에서
                # 재확인 결과로 다시 계산해 덮거나 지운다(#148). 1차 fix는 방금
                # 실행해 성공했을 수도 있는 명령이라 그대로 두면 끝난 일을 다시
                # 안내하게 된다.
                carried = {
                    key: fix_target[key]
                    for key in ("model_name", "batch_size", "seq_len", "fix")
                    if key in fix_target
                }
                # `list.index()`는 dict를 값으로 비교하므로 내용이 같은 다른 항목을
                # 집을 수 있다 — 갈아끼울 자리는 동일성으로 찾는다.
                index = next(i for i, r in enumerate(results) if r is fix_target)
                reverified_result = {
                    **reverified,
                    **carried,
                    "reverified": True,
                    # 1차 판정을 함께 실어 보낸다. 교체해버리면 화면에 "수정 후"만
                    # 남아, --yes가 실제로 무엇을 바꿨는지가 사라진다(#88).
                    "previous_verdict": fix_target.get("verdict"),
                }
                # **fix는 재확인 결과로 다시 계산한다.** `carried`가 옮겨온 1차 fix는
                # 방금 실행해서 **성공했을 수도 있는** 명령이다. 그대로 두면 화면이
                # 이미 끝난 일을 "지금 하세요"로 다시 안내한다 — 원래 원인은 고쳐졌는데
                # 다른 이유로 여전히 PASS가 아닐 때 정확히 어긋난다. 실제로 torch를
                # 재설치해 device=cuda가 된 뒤에도 같은 torch 재설치 명령이 다시
                # 안내됐다(#148, 상영님 실측). 남은 문제(bnb·peft 미설치)의 명령은
                # 화면에 없었다.
                #
                # "무엇을 실행했는가"라는 기록은 아래 `자동 수정 실행: <command>`
                # 알림이 담당하므로(--json의 notices에도 그대로 실린다) 잃는 정보가 없다.
                refreshed_fix = suggest_fix(reverified_result)
                if refreshed_fix:
                    reverified_result["fix"] = refreshed_fix
                else:
                    # 재확인이 PASS면 suggest_fix가 None을 준다. 1차 fix를 남겨두면
                    # 다 고쳐진 화면에 수정 안내가 붙는다.
                    reverified_result.pop("fix", None)
                results[index] = reverified_result
                # 재확인 1건으로 전체 verdict를 대체하지 않는다 — 그러면 재확인하지
                # 않은 나머지 결과(예: 기본 체크의 WARN)가 종료 코드에서 사라진다(#68).
                verdict = _aggregate_verdict(results)

                # 기본 체크 재확인으로 FAIL이 풀리면, fail-fast로 생략됐던 모델
                # 체크를 이어서 실행한다. 안 그러면 모델을 한 번도 확인한 적이
                # 없는데 화면과 종료 코드는 "이상 없음"이라고 말한다(#84) —
                # --model과 --yes를 함께 쓴 사용자가 물은 질문("이 모델이 이
                # 기계에서 도는가")에 끝내 답하지 않은 채로 성공 취급되는 셈이다.
                # 생략은 기본 체크가 FAIL일 때만 일어나므로(cli.md "결과 집계"),
                # fix_target이 기본 체크(model_name 없음)일 때만 해당한다.
                if fix_target.get("model_name") is None and results[index]["verdict"] != "FAIL":
                    skipped_index = next((i for i, r in enumerate(results) if "skipped" in r), None)
                    if skipped_index is not None:
                        skipped = results[skipped_index]
                        model_name = skipped["model_name"]
                        model_batch_size = skipped.get("batch_size", 1)
                        model_seq_len = skipped.get("seq_len", 8)
                        # 기본 체크 재확인과 같은 이유로 지금 시점의 GPU 상태를
                        # 다시 조회한다 — fix 실행 전 값은 더 이상 유효하지 않다
                        # (reverify.py 참고).
                        model_state = query_gpu_state()
                        _progress("진단 중… (torch 불러오기 · canary 실행, 수 초~수십 초)")
                        raw_model = run_canary_check(
                            model_name=model_name,
                            batch_size=model_batch_size,
                            seq_len=model_seq_len,
                        )
                        if model_state:
                            raw_model["env"] = {
                                **(raw_model.get("env") or {}),
                                "gpu_free_mb": model_state["free_mb"],
                                "gpu_total_mb": model_state["total_mb"],
                                "gpu_driver_version": model_state["driver_version"],
                                "gpu_name": model_state["name"],
                            }
                        model_res = judge_result(raw_model)
                        results[skipped_index] = {
                            **model_res,
                            "model_name": model_name,
                            "batch_size": model_batch_size,
                            "seq_len": model_seq_len,
                        }
                        notices.append(
                            "--yes: 기본 체크 통과로 생략됐던 모델 체크"
                            f"({model_name})를 이어서 실행했다"
                        )
                        verdict = _aggregate_verdict(results)

    # **`results`가 최종 확정된 뒤에 판단한다.** `--yes`로 기본 체크를 고치면
    # 생략됐던 모델 체크를 이어서 실행하는 경로가 있어(#84), 앞에서 판단하면
    # 그때 실제로 돈 모델 체크에는 안내가 빠진다 — VRAM 부족 FAIL을 보여주면서
    # 그게 vanilla 기준이라는 걸 안 알려주는, 이 안내가 막으려던 바로 그 상황이
    # 그 경로에만 남는다(#118, PR 리뷰에서 성오님 지적).
    if _model_check_ran(results):
        notices.append(_VANILLA_PATH_NOTICE)

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
