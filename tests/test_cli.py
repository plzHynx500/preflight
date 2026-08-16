from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from preflight.cli import app, get_exit_code

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
        patch("preflight.cli.render_report"),
    ):
        result = runner.invoke(
            app,
            [
                "check",
            ],
        )
        assert result.exit_code == 0


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
        assert model_call_arg["env"]["gpu_free_mb"] == 10000
        assert model_call_arg["env"]["gpu_total_mb"] == 12000
