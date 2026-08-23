from __future__ import annotations

import io
import re
import subprocess
import sys
from unittest.mock import patch

from typer.testing import CliRunner

from preflight import __version__
from preflight.cli import _select_fix_target, app, ensure_utf8_streams, get_exit_code
from preflight.report import render_report

runner = CliRunner()


def _make_stderr_runner() -> CliRunner:
    """`result.stderr`를 쓸 수 있는 CliRunner를 만든다.

    CI 매트릭스의 Python 3.9는 typer 0.23 + 외부 click 8.1을 받는데, 그 click의
    `CliRunner`는 `mix_stderr=False`를 명시해야 stdout/stderr가 분리된다(기본은
    합쳐서 `stdout`에만 담긴다). Python 3.12(로컬 개발환경)는 typer 0.27+가 click을
    자체 벤더링하고 있어 `mix_stderr` 파라미터 자체가 없고 항상 분리돼 있다 —
    그래서 버전에 따라 있는 파라미터가 다르다.
    """
    try:
        return CliRunner(mix_stderr=False)
    except TypeError:
        return CliRunner()


stderr_runner = _make_stderr_runner()

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """rich가 넣는 색상 코드를 지운다.

    강제 컬러 환경(GitHub Actions CI 등)에서는 rich가 `--batch-size`처럼 하이픈으로
    나뉜 옵션 이름을 토큰별로 따로 스타일링해 `-`·`-batch`·`-size` 사이에 ANSI
    코드를 끼워 넣는다 — 로컬(색상 미강제)에서는 코드가 안 붙어 통과하던 substring
    검사가 CI에서만 깨졌다(#59 PR 실측). 코드만 제거하면 문자는 그대로 이어 붙는다.
    """
    return _ANSI_ESCAPE.sub("", text)


def test_check_command_exists() -> None:
    result = runner.invoke(app, ["check", "--help"])
    assert result.exit_code == 0


def test_version_flag_prints_version_and_exits() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.output


def test_help_does_not_leak_internal_docs_path() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "docs/contracts" not in result.output


def test_check_help_does_not_leak_internal_docs_path() -> None:
    result = runner.invoke(app, ["check", "--help"])

    assert result.exit_code == 0
    assert "docs/contracts" not in result.output


def test_get_exit_code() -> None:
    assert get_exit_code("PASS") == 0
    assert get_exit_code("FAIL") == 1
    assert get_exit_code("WARN") == 2
    assert get_exit_code("pass") == 0
    assert get_exit_code("warn") == 2
    assert get_exit_code("fail") == 1
    assert get_exit_code("") == 0


def test_check_exit_code_pass() -> None:
    fake_raw = {"status": "ok", "device": "cuda"}
    fake_res = {
        "status": "ok",
        "device": "cuda",
        "verdict": "PASS",
        "reasons": [],
    }
    with (
        patch("preflight.cli.run_canary_check", return_value=fake_raw),
        patch("preflight.cli.judge_result", return_value=fake_res),
        patch("preflight.cli.render_report") as mock_render,
    ):
        result = runner.invoke(
            app,
            [
                "check",
            ],
        )
        assert result.exit_code == 0

        mock_render.assert_called_once()
        kwargs = mock_render.call_args.kwargs
        assert "elapsed_seconds" in kwargs
        assert isinstance(kwargs["elapsed_seconds"], float)
        assert kwargs["elapsed_seconds"] > 0


def test_check_exit_code_fail() -> None:
    fake_raw = {"status": "import_crash", "error_log": "CUDA error"}
    fake_res = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["import_crash"],
        "error_log": "CUDA error",
    }
    with (
        patch("preflight.cli.run_canary_check", return_value=fake_raw),
        patch("preflight.cli.judge_result", return_value=fake_res),
        patch("preflight.cli.render_report"),
    ):
        result = runner.invoke(
            app,
            [
                "check",
            ],
        )
        assert result.exit_code == 1


def test_check_exit_code_warn() -> None:
    fake_raw = {"status": "ok", "device": "cuda"}
    fake_res = {
        "status": "ok",
        "device": "cuda",
        "verdict": "WARN",
        "reasons": ["cpu_multiplier_low"],
    }
    with (
        patch("preflight.cli.run_canary_check", return_value=fake_raw),
        patch("preflight.cli.judge_result", return_value=fake_res),
        patch("preflight.cli.render_report"),
    ):
        result = runner.invoke(
            app,
            [
                "check",
            ],
        )
        assert result.exit_code == 2


def test_check_exit_code_yes_reverify_pass() -> None:
    # error_log에 bitsandbytes 시그니처가 있어야 fix_argv가 붙어 reverify 경로를 탄다(#51).
    fake_raw = {"status": "import_crash", "error_log": "libbitsandbytes_cpu.so: CUDA error"}
    fake_initial = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["import_crash"],
        "error_log": "libbitsandbytes_cpu.so: CUDA error",
    }
    fake_reverified = {
        "status": "ok",
        "device": "cuda",
        "verdict": "PASS",
        "reasons": [],
    }
    with (
        patch("preflight.cli.run_canary_check", return_value=fake_raw),
        patch("preflight.cli.judge_result", return_value=fake_initial),
        patch("preflight.cli.apply_fix"),
        patch("preflight.cli.reverify", return_value=fake_reverified),
        patch("preflight.cli.render_report"),
    ):
        result = runner.invoke(app, ["check", "--yes"])
        assert result.exit_code == 0


def test_check_exit_code_yes_reverify_warn() -> None:
    # error_log에 bitsandbytes 시그니처가 있어야 fix_argv가 붙어 reverify 경로를 탄다(#51).
    fake_raw = {"status": "import_crash", "error_log": "libbitsandbytes_cpu.so: CUDA error"}
    fake_initial = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["import_crash"],
        "error_log": "libbitsandbytes_cpu.so: CUDA error",
    }
    fake_reverified = {
        "status": "ok",
        "device": "cuda",
        "verdict": "WARN",
        "reasons": ["cpu_multiplier_low"],
    }
    with (
        patch("preflight.cli.run_canary_check", return_value=fake_raw),
        patch("preflight.cli.judge_result", return_value=fake_initial),
        patch("preflight.cli.apply_fix"),
        patch("preflight.cli.reverify", return_value=fake_reverified),
        patch("preflight.cli.render_report"),
    ):
        result = runner.invoke(app, ["check", "--yes"])
        assert result.exit_code == 2


def test_check_exit_code_yes_reverify_fail() -> None:
    fake_raw = {"status": "import_crash", "error_log": "CUDA error"}
    fake_initial = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["import_crash"],
        "error_log": "CUDA error",
    }
    fake_reverified = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["import_crash"],
        "error_log": "CUDA error",
    }
    with (
        patch("preflight.cli.run_canary_check", return_value=fake_raw),
        patch("preflight.cli.judge_result", return_value=fake_initial),
        patch("preflight.cli.apply_fix"),
        patch("preflight.cli.reverify", return_value=fake_reverified),
        patch("preflight.cli.render_report"),
    ):
        result = runner.invoke(app, ["check", "--yes"])
        assert result.exit_code == 1


def test_check_yes_reverify_runs_previously_skipped_model_check() -> None:
    """기본 체크가 FAIL → 모델 체크 생략 → --yes로 기본 체크가 PASS로 뒤집히면,
    생략됐던 모델 체크를 이어서 실행하고 그 판정까지 exit code에 반영한다(#84).

    수정 전에는 기본 체크만 재확인하고 모델 체크는 skipped로 남아, 모델을 한 번도
    확인한 적이 없는데 exit code가 0이 되는 문제가 있었다.
    """
    fake_raw_basic = {"status": "import_crash", "error_log": "libbitsandbytes_cpu.so: CUDA error"}
    fake_basic_initial = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["import_crash"],
        "error_log": "libbitsandbytes_cpu.so: CUDA error",
    }
    fake_reverified = {"status": "ok", "device": "cuda", "verdict": "PASS", "reasons": []}
    fake_raw_model = {"status": "oom"}
    # 이어서 실행된 모델 체크가 FAIL이면, 기본 체크가 PASS로 뒤집혔더라도 최종
    # exit code는 그 FAIL을 반영해야 한다 — 그러지 않으면 이 이슈가 재발한다.
    fake_model_res = {
        "status": "oom",
        "device": "cuda",
        "verdict": "FAIL",
        "reasons": ["status_oom"],
    }

    with (
        patch("preflight.cli.query_gpu_state", return_value=None),
        patch(
            "preflight.cli.run_canary_check", side_effect=[fake_raw_basic, fake_raw_model]
        ) as mock_run,
        patch(
            "preflight.cli.judge_result", side_effect=[fake_basic_initial, fake_model_res]
        ) as mock_judge,
        patch("preflight.cli.apply_fix"),
        patch("preflight.cli.reverify", return_value=fake_reverified),
        patch("preflight.cli.render_report") as mock_render,
    ):
        result = runner.invoke(app, ["check", "--model", "dummy/model", "--yes"])

    assert mock_run.call_count == 2
    assert mock_judge.call_count == 2
    assert result.exit_code == 1

    results = mock_render.call_args.args[0]
    assert "skipped" not in results[1]
    assert results[1]["verdict"] == "FAIL"
    assert results[1]["model_name"] == "dummy/model"

    notices = mock_render.call_args.kwargs["notices"]
    assert any("이어서 실행" in notice for notice in notices)


def test_check_yes_reverify_still_fail_leaves_model_check_skipped() -> None:
    """기본 체크 재확인이 여전히 FAIL이면 모델 체크는 실행하지 않는다(#84).

    수십 초가 드는 모델 canary를 어차피 fail-fast로 다시 생략될 상황에서까지
    돌리지 않는다.
    """
    fake_raw_basic = {"status": "import_crash", "error_log": "libbitsandbytes_cpu.so: CUDA error"}
    fake_basic_initial = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["import_crash"],
        "error_log": "libbitsandbytes_cpu.so: CUDA error",
    }
    fake_reverified = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["import_crash"],
        "error_log": "libbitsandbytes_cpu.so: CUDA error",
    }

    with (
        patch("preflight.cli.query_gpu_state", return_value=None),
        patch("preflight.cli.run_canary_check", side_effect=[fake_raw_basic]) as mock_run,
        patch("preflight.cli.judge_result", side_effect=[fake_basic_initial]) as mock_judge,
        patch("preflight.cli.apply_fix"),
        patch("preflight.cli.reverify", return_value=fake_reverified),
        patch("preflight.cli.render_report") as mock_render,
    ):
        result = runner.invoke(app, ["check", "--model", "dummy/model", "--yes"])

    assert mock_run.call_count == 1
    assert mock_judge.call_count == 1
    assert result.exit_code == 1

    results = mock_render.call_args.args[0]
    assert results[1].get("skipped") == "환경 체크 실패"

    notices = mock_render.call_args.kwargs["notices"]
    assert not any("이어서 실행" in notice for notice in notices)


def test_cli_check_with_model_injects_gpu_state() -> None:
    fake_state = {
        "free_mb": 10000,
        "total_mb": 12000,
        "driver_version": "560.94.03",
        "name": "NVIDIA GeForce RTX 4070 Ti",
    }
    fake_raw_basic = {"status": "ok", "env": {"dummy": 1}}
    fake_raw_model = {"status": "ok", "env": {"dummy": 2}}

    fake_basic_res = {"verdict": "PASS"}
    fake_model_res = {"verdict": "PASS"}

    def mock_run_canary_check(model_name, **kwargs):
        if model_name is None:
            return fake_raw_basic.copy()
        return fake_raw_model.copy()

    with (
        patch("preflight.cli.query_gpu_state", return_value=fake_state),
        patch("preflight.cli.run_canary_check", side_effect=mock_run_canary_check),
        patch(
            "preflight.cli.judge_result", side_effect=[fake_basic_res, fake_model_res]
        ) as mock_judge,
        patch("preflight.cli.render_report"),
    ):
        result = runner.invoke(app, ["check", "--model", "dummy/model"])
        assert result.exit_code == 0

        assert mock_judge.call_count == 2
        model_call_arg = mock_judge.call_args_list[1][0][0]
        assert "env" in model_call_arg
        assert model_call_arg["env"]["dummy"] == 2
        assert model_call_arg["env"]["gpu_free_mb"] == 10000
        assert model_call_arg["env"]["gpu_total_mb"] == 12000
        assert model_call_arg["env"]["gpu_driver_version"] == "560.94.03"
        assert model_call_arg["env"]["gpu_name"] == "NVIDIA GeForce RTX 4070 Ti"


def test_fix_attaches_to_fail_not_to_earlier_warn() -> None:
    """FIX는 결과 배열 순서가 아니라 심각도를 보고 붙는다 (#69).

    기본 체크 WARN(cpu_multiplier_low) + 모델 체크 FAIL(status_oom)에서, 예전에는
    처음 만난 non-PASS인 WARN에 FIX가 붙었다 — 사용자는 화면의 OOM FAIL을 보면서
    "CPU 대비 연산 속도 2배 미만" 안내를 받고, 정작 batch_size 축소 안내는
    어디에도 나오지 않았다.
    """
    basic_warn = {
        "status": "ok",
        "device": "cuda",
        "verdict": "WARN",
        "reasons": ["cpu_multiplier_low"],
    }
    model_fail = {
        "status": "oom",
        "device": "cuda",
        "verdict": "FAIL",
        "reasons": ["status_oom"],
    }

    with (
        patch("preflight.cli.query_gpu_state", return_value=None),
        patch("preflight.cli.run_canary_check", return_value={"status": "ok"}),
        patch("preflight.cli.judge_result", side_effect=[basic_warn, model_fail]),
        patch("preflight.cli.render_report") as mock_render,
    ):
        result = runner.invoke(
            app, ["check", "--model", "dummy/model", "--batch-size", "2", "--seq-len", "2048"]
        )

    assert result.exit_code == 1
    results = mock_render.call_args[0][0]
    assert "fix" not in results[0]
    assert results[1]["fix"]["cause"] == "oom"
    assert "batch_size" in results[1]["fix"]["message"]


def test_select_fix_target_prefers_first_fail() -> None:
    """FAIL이 여럿이면 그중 첫 번째에 붙는다 — 현재 동작 유지 (#69).

    지금 파이프라인에서는 기본 체크가 FAIL이면 모델 체크가 fail-fast로 생략돼
    FAIL이 둘 나올 수 없으므로, 규칙 자체를 함수 단위로 확인한다.
    """
    first = {"status": "import_crash", "verdict": "FAIL", "reasons": ["status_import_crash"]}
    second = {"status": "oom", "verdict": "FAIL", "reasons": ["status_oom"]}
    warn = {"status": "ok", "verdict": "WARN", "reasons": ["cpu_multiplier_low"]}

    results = [warn, first, second]
    target = _select_fix_target(results)

    assert target is first
    assert first["fix"]["cause"] == "import_crash_general"
    assert "fix" not in warn
    assert "fix" not in second


def test_batch_size_zero_is_rejected_by_parser() -> None:
    """`--batch-size 0`은 canary를 돌리기 전에 파서가 거부한다 (#59).

    막지 않으면 0이 그대로 worker까지 흘러가 `torch.randint(..., (0, 8))` 같은 빈
    텐서로 forward/backward가 "성공"해버려 status="ok"(PASS)가 나가는 거짓 양성이
    된다.
    """
    result = runner.invoke(app, ["check", "--model", "dummy/model", "--batch-size", "0"])

    assert result.exit_code != 0
    assert "batch-size" in _strip_ansi(result.output)


def test_batch_size_negative_is_rejected_by_parser() -> None:
    """`--batch-size -1`도 같은 이유로 거부된다 (#59)."""
    result = runner.invoke(app, ["check", "--model", "dummy/model", "--batch-size", "-1"])

    assert result.exit_code != 0
    assert "batch-size" in _strip_ansi(result.output)


def test_seq_len_zero_is_rejected_by_parser() -> None:
    """`--seq-len 0`도 같은 이유로 거부된다 (#59)."""
    result = runner.invoke(app, ["check", "--model", "dummy/model", "--seq-len", "0"])

    assert result.exit_code != 0
    assert "seq-len" in _strip_ansi(result.output)


def test_top_level_help_survives_redirect() -> None:
    """`preflight --help`를 리다이렉트해도 죽지 않는다 (#89).

    `ensure_utf8_streams()`가 `@app.callback()` 안에 있으면 Click이 그룹의
    `--help`를 콜백 실행 전에 처리하고 종료해버려, 콜백을 한 번도 안 거친 채로
    rich 도움말 상자가 로케일 인코딩(cp949 등)에서 죽는다. CliRunner는 캡처용
    가짜 스트림을 쓰기 때문에 이 문제를 재현하지 못하므로, 실제 자식 프로세스로
    `--help`를 파일로 리다이렉트해 재현해야 한다. `check --help`처럼 서브커맨드
    `--help`는 그룹 콜백이 먼저 돌아 우연히 살아남으므로 이 회귀를 못 잡는다 —
    반드시 최상위 `--help`여야 한다.
    """
    result = subprocess.run(
        [sys.executable, "-c", "from preflight.cli import app; app()", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert result.returncode == 0
    assert "UnicodeEncodeError" not in result.stderr
    assert result.stderr == ""


def test_ensure_utf8_streams_survives_cp949_output(monkeypatch) -> None:
    """cp949로 리다이렉트된 stdout에도 ✖·— 같은 기호를 죽지 않고 찍는다 (#54).

    Windows에서 `preflight check > file`처럼 stdout이 파이프/파일로 바뀌면
    인코딩이 로케일(cp949 등)을 따라가 rich 콘솔의 ✔/✖/⚠/…/— 기호에서
    UnicodeEncodeError로 죽었다. ensure_utf8_streams()가 stdout을 UTF-8로
    재설정한 뒤에는 render_report()가 문제없이 완료돼야 한다.
    """
    buffer = io.BytesIO()
    cp949_stdout = io.TextIOWrapper(buffer, encoding="cp949")
    monkeypatch.setattr("sys.stdout", cp949_stdout)

    ensure_utf8_streams()

    results = [
        {
            "status": "import_crash",
            "verdict": "FAIL",
            "reasons": ["status_import_crash"],
            "error_log": "CUDA error",
        },
        {
            "model_name": "dummy/model",
            "batch_size": 1,
            "seq_len": 8,
            "skipped": "환경 체크 실패",
        },
    ]

    render_report(results, json_output=False, elapsed_seconds=1.0)
    cp949_stdout.flush()

    output = buffer.getvalue().decode("utf-8")
    assert "✖" in output
    assert "—" in output


def test_check_prints_progress_to_stderr_before_canary_runs() -> None:
    """canary 시작 직전 진단 중 안내가 stderr에 찍힌다 (#63).

    기본 체크만 첫 줄까지 10초 안팎 아무 출력이 없어 멈춘 것처럼 보였다 — canary를
    부르기 전에 stderr로 한 줄 안내해 화면이 비어있지 않게 한다.
    """
    fake_raw = {"status": "ok", "device": "cuda"}
    fake_res = {"status": "ok", "device": "cuda", "verdict": "PASS", "reasons": []}
    with (
        patch("preflight.cli.run_canary_check", return_value=fake_raw),
        patch("preflight.cli.judge_result", return_value=fake_res),
        patch("preflight.cli.render_report"),
    ):
        result = stderr_runner.invoke(app, ["check"])

        assert result.exit_code == 0
        assert "진단 중" in result.stderr


def test_check_json_stdout_stays_pure_while_progress_goes_to_stderr() -> None:
    """`--json`의 stdout 순수성은 stderr 진행 표시와 무관하게 유지된다 (#63).

    stdout이 파일로 리다이렉트되는 CI 연동 시나리오(`preflight check --json >
    result.json`)에서, 진행 표시가 stdout에 섞이면 JSON 파싱이 깨진다.
    """
    import json as json_module

    fake_raw = {"status": "ok", "device": "cuda"}
    fake_res = {"status": "ok", "device": "cuda", "verdict": "PASS", "reasons": []}
    with (
        patch("preflight.cli.run_canary_check", return_value=fake_raw),
        patch("preflight.cli.judge_result", return_value=fake_res),
    ):
        result = stderr_runner.invoke(app, ["check", "--json"])

        assert result.exit_code == 0
        assert "진단 중" in result.stderr
        payload = json_module.loads(result.stdout)
        assert payload["summary"]["fail"] == 0


def test_check_yes_prints_fix_and_reverify_progress_to_stderr_in_order() -> None:
    """`--yes`가 수정 명령 실행·재확인 직전 각각 안내를 stderr에 찍는다 (#63).

    `--yes`는 pip 재설치 + canary 재실행으로 1분 넘게 침묵한다 — 두 단계 경계에
    안내가 없으면 어디까지 진행됐는지 알 수 없다. 두 줄이 이 순서로 나와야 한다.
    """
    fake_raw = {"status": "import_crash", "error_log": "libbitsandbytes_cpu.so: CUDA error"}
    fake_initial = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["import_crash"],
        "error_log": "libbitsandbytes_cpu.so: CUDA error",
    }
    fake_reverified = {"status": "ok", "device": "cuda", "verdict": "PASS", "reasons": []}
    with (
        patch("preflight.cli.run_canary_check", return_value=fake_raw),
        patch("preflight.cli.judge_result", return_value=fake_initial),
        patch("preflight.cli.apply_fix"),
        patch("preflight.cli.reverify", return_value=fake_reverified),
        patch("preflight.cli.render_report"),
    ):
        result = stderr_runner.invoke(app, ["check", "--yes"])

        assert result.exit_code == 0
        fix_index = result.stderr.index("수정 명령 실행 중")
        reverify_index = result.stderr.index("재확인 중")
        assert fix_index < reverify_index


def test_check_yes_no_fix_command_skips_fix_progress_line() -> None:
    """실행할 수정 명령이 없으면 "수정 명령 실행 중" 줄도 찍지 않는다 (#63).

    `fix_argv`가 없는 원인(예: import_crash_general)은 apply_fix를 아예 부르지
    않으므로, 부르지도 않을 단계를 진행 중이라고 안내하면 거짓 정보가 된다.
    """
    fake_raw = {"status": "import_crash", "error_log": "이상한 오류"}
    fake_initial = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["import_crash"],
        "error_log": "이상한 오류",
    }
    with (
        patch("preflight.cli.run_canary_check", return_value=fake_raw),
        patch("preflight.cli.judge_result", return_value=fake_initial),
        patch("preflight.cli.render_report"),
    ):
        result = stderr_runner.invoke(app, ["check", "--yes"])

        assert "수정 명령 실행 중" not in result.stderr
        assert "재확인 중" not in result.stderr


def test_python_dash_m_preflight_runs() -> None:
    """`python -m preflight`가 콘솔 스크립트와 같이 동작한다 (#64).

    Windows에서 venv를 활성화하지 않았거나 `pip install --user`로 설치하면
    `Scripts/`가 PATH에 없어 `preflight` 명령을 못 찾는다. 그때의 표준 폴백이
    이 호출인데, `__main__.py`가 없으면 "cannot be directly executed"로 막혔다.

    출력을 캡처(파이프)해서 부르므로 #89(최상위 --help가 리다이렉트 시
    UnicodeEncodeError로 죽던 문제)의 회귀 가드도 겸한다 — 그 버그가 되살아나면
    여기서 종료 코드가 1이 된다.

    도움말 **문구**에는 단정을 걸지 않는다. rich가 그리는 화면은 typer/click
    버전마다 달라서 CI에서만 깨진다(PR #90 리뷰, 상영님 실측: CI click 8.1.8 vs
    로컬 8.4.2). 여기서 확인할 것은 모듈이 해석돼 정상 종료하는가다.
    """
    result = subprocess.run(
        [sys.executable, "-m", "preflight", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "cannot be directly executed" not in result.stderr
    assert "UnicodeEncodeError" not in result.stderr
    assert "check" in _strip_ansi(result.stdout)
    assert "cannot be directly executed" not in result.stderr
