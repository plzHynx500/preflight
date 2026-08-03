from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from preflight.cli import app
from preflight.fix.causes import classify_cause
from preflight.fix.executor import FixExecutionError, apply_fix, suggest_fix


def test_suggest_fix_none_when_pass() -> None:
    check_result = {
        "status": "ok",
        "device": "cuda",
        "verdict": "PASS",
        "reasons": [],
    }
    assert classify_cause(check_result) == "pass"
    assert suggest_fix(check_result) is None


def test_classify_and_suggest_bnb_not_compiled() -> None:
    # case 1: import_crash with bitsandbytes/CUDA log
    res1 = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["import_crash"],
        "error_log": "libbitsandbytes_cpu.so CUDA Setup failed",
    }
    assert classify_cause(res1) == "bnb_not_compiled_with_cuda"
    fix1 = suggest_fix(res1)
    assert fix1 is not None
    assert fix1["cause"] == "bnb_not_compiled_with_cuda"
    assert fix1["fix_command"] == "pip install bitsandbytes --upgrade --force-reinstall"

    # case 2: 4bit cpu fallback with compiled_with_cuda=False
    res2 = {
        "status": "ok",
        "verdict": "FAIL",
        "reasons": ["4bit_cpu_fallback"],
        "compiled_with_cuda": False,
    }
    assert classify_cause(res2) == "bnb_not_compiled_with_cuda"
    fix2 = suggest_fix(res2)
    assert fix2 is not None
    assert fix2["cause"] == "bnb_not_compiled_with_cuda"
    assert fix2["fix_command"] == "pip install bitsandbytes --upgrade --force-reinstall"


def test_classify_and_suggest_import_general() -> None:
    res = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["import_crash"],
        "error_log": "ModuleNotFoundError: No module named 'custom_mod'",
    }
    assert classify_cause(res) == "import_crash_general"
    fix = suggest_fix(res)
    assert fix is not None
    assert fix["cause"] == "import_crash_general"
    assert fix["fix_command"] is None


def test_classify_and_suggest_4bit_cpu_other() -> None:
    res = {
        "status": "ok",
        "verdict": "FAIL",
        "reasons": ["4bit_cpu_fallback"],
        "compiled_with_cuda": True,
    }
    assert classify_cause(res) == "4bit_cpu_fallback_other"
    fix = suggest_fix(res)
    assert fix is not None
    assert fix["cause"] == "4bit_cpu_fallback_other"
    assert fix["fix_command"] is None


def test_classify_and_suggest_oom() -> None:
    res = {
        "status": "oom",
        "verdict": "FAIL",
        "reasons": ["oom"],
    }
    assert classify_cause(res) == "oom"
    fix = suggest_fix(res)
    assert fix is not None
    assert fix["cause"] == "oom"
    assert fix["fix_command"] is None
    assert "batch_size" in fix["message"]


def test_classify_and_suggest_warn_grey_zones() -> None:
    # memory_delta_high
    res_mem = {
        "status": "ok",
        "verdict": "WARN",
        "reasons": ["memory_delta_high"],
    }
    assert classify_cause(res_mem) == "memory_delta_high"
    fix_mem = suggest_fix(res_mem)
    assert fix_mem is not None
    assert fix_mem["cause"] == "memory_delta_high"
    assert fix_mem["fix_command"] is None

    # cpu_multiplier_low
    res_cpu = {
        "status": "ok",
        "verdict": "WARN",
        "reasons": ["cpu_multiplier_low"],
    }
    assert classify_cause(res_cpu) == "cpu_multiplier_low"
    fix_cpu = suggest_fix(res_cpu)
    assert fix_cpu is not None
    assert fix_cpu["cause"] == "cpu_multiplier_low"
    assert fix_cpu["fix_command"] is None


def test_priority_fail_over_warn() -> None:
    res = {
        "status": "oom",
        "verdict": "FAIL",
        "reasons": ["oom", "memory_delta_high", "cpu_multiplier_low"],
    }
    assert classify_cause(res) == "oom"
    fix = suggest_fix(res)
    assert fix is not None
    assert fix["cause"] == "oom"


def test_apply_fix_none_or_empty_command() -> None:
    with patch("subprocess.run") as mock_run:
        apply_fix({"fix_command": None})
        apply_fix({"fix_command": ""})
        mock_run.assert_not_called()


def test_apply_fix_executes_command_successfully() -> None:
    with patch("subprocess.run") as mock_run:
        apply_fix({"fix_command": "pip install bitsandbytes --upgrade"})
        mock_run.assert_called_once_with(
            ["pip", "install", "bitsandbytes", "--upgrade"],
            capture_output=True,
            text=True,
            check=True,
        )


def test_apply_fix_raises_fix_execution_error_on_called_process_error() -> None:
    err = subprocess.CalledProcessError(
        returncode=1,
        cmd=["pip", "install", "foo"],
        output="stdout details",
        stderr="stderr details",
    )
    with patch("subprocess.run", side_effect=err):
        with pytest.raises(FixExecutionError) as exc_info:
            apply_fix({"fix_command": "pip install foo"})
        assert exc_info.value.returncode == 1
        assert "stderr details" in exc_info.value.stderr
        assert "stdout details" in exc_info.value.stdout


def test_apply_fix_raises_fix_execution_error_on_os_error() -> None:
    with patch("subprocess.run", side_effect=OSError("No such file")):
        with pytest.raises(FixExecutionError) as exc_info:
            apply_fix({"fix_command": "pip install foo"})
        assert exc_info.value.returncode is None
        assert "No such file" in str(exc_info.value)


def test_cli_check_without_yes_does_not_execute_fix() -> None:
    runner = CliRunner()
    fake_raw = {"status": "import_crash", "error_log": "CUDA Setup failed"}
    fake_res = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["import_crash"],
        "error_log": "CUDA Setup failed",
    }

    with (
        patch("preflight.cli.run_canary_check", return_value=fake_raw),
        patch("preflight.cli.judge_result", return_value=fake_res),
        patch("preflight.cli.render_report"),
        patch("preflight.cli.apply_fix") as mock_apply_fix,
    ):
        result = runner.invoke(app, [])
        assert result.exit_code == 1
        mock_apply_fix.assert_not_called()


def test_cli_check_with_yes_executes_fix() -> None:
    runner = CliRunner()
    fake_raw = {"status": "import_crash", "error_log": "CUDA Setup failed"}
    fake_res = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["import_crash"],
        "error_log": "CUDA Setup failed",
    }

    with (
        patch("preflight.cli.run_canary_check", return_value=fake_raw),
        patch("preflight.cli.judge_result", return_value=fake_res),
        patch("preflight.cli.render_report"),
        patch("preflight.cli.apply_fix") as mock_apply_fix,
    ):
        result = runner.invoke(app, ["--yes"])
        assert result.exit_code == 1
        mock_apply_fix.assert_called_once()
        fix_arg = mock_apply_fix.call_args[0][0]
        assert fix_arg["cause"] == "bnb_not_compiled_with_cuda"
        assert fix_arg["fix_command"] == "pip install bitsandbytes --upgrade --force-reinstall"


def test_cli_check_with_yes_no_fix_when_pass() -> None:
    runner = CliRunner()
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
        patch("preflight.cli.apply_fix") as mock_apply_fix,
    ):
        result = runner.invoke(app, ["--yes"])
        assert result.exit_code == 0
        mock_apply_fix.assert_not_called()
