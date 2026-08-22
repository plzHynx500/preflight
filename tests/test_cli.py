from __future__ import annotations

import io
import subprocess
import sys
from unittest.mock import patch

from typer.testing import CliRunner

from preflight.cli import _select_fix_target, app, ensure_utf8_streams, get_exit_code
from preflight.report import render_report

runner = CliRunner()


def test_check_command_exists() -> None:
    result = runner.invoke(app, ["check", "--help"])
    assert result.exit_code == 0


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


def test_cli_check_with_model_injects_gpu_state() -> None:
    fake_state = {"free_mb": 10000, "total_mb": 12000}
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
