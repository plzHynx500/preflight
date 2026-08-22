from __future__ import annotations

import io
import subprocess
import sys
from unittest.mock import patch

from typer.testing import CliRunner

from preflight.cli import app, ensure_utf8_streams, get_exit_code
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
    fake_raw = {"status": "import_crash", "error_log": "CUDA error"}
    fake_initial = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["import_crash"],
        "error_log": "CUDA error",
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
    fake_raw = {"status": "import_crash", "error_log": "CUDA error"}
    fake_initial = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["import_crash"],
        "error_log": "CUDA error",
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


def test_python_dash_m_preflight_runs() -> None:
    """`python -m preflight`가 콘솔 스크립트와 같이 동작한다 (#64).

    Windows에서 venv를 활성화하지 않았거나 `pip install --user`로 설치하면
    `Scripts/`가 PATH에 없어 `preflight` 명령을 못 찾는다. 그때의 표준 폴백이
    이 호출인데, `__main__.py`가 없으면 "cannot be directly executed"로 막혔다.

    콘솔 스크립트(`preflight`)는 설치된 환경에서만 존재하므로, 어디서나 도는
    모듈 실행 경로만 검증한다.
    """
    result = subprocess.run(
        # 최상위 --help가 아니라 서브커맨드를 쓴다 — 최상위 --help는 파이프로 보낼 때
        # 별개 버그로 죽는다(#89). 여기서 확인할 것은 모듈 해석이 되는가다.
        [sys.executable, "-m", "preflight", "check", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--model" in result.stdout
    assert "cannot be directly executed" not in result.stderr
