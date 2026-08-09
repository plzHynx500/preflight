"""ReVerifier 단위 및 CLI 연동 통합 테스트 (W11: FR-06).

docs/architecture.md §5 MODULE-03 참고.
"""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from preflight.cli import app
from preflight.fix.executor import FixExecutionError
from preflight.reverify import reverify

runner = CliRunner()


def test_reverify_calls_engine_and_judge() -> None:
    fake_raw = {
        "status": "ok",
        "device": "cuda",
        "memory_delta_mb": 100.0,
        "elapsed_ms": 10.0,
        "cpu_multiplier": 30.0,
        "quant_backend": "bnb-4bit",
        "error_log": None,
    }
    fake_judged = {
        **fake_raw,
        "verdict": "PASS",
        "reasons": [],
    }

    with (
        patch("preflight.reverify.run_canary_check", return_value=fake_raw) as mock_engine,
        patch("preflight.reverify.judge_result", return_value=fake_judged) as mock_judge,
    ):
        res = reverify("meta-llama/Llama-3.1-8B", batch_size=2, seq_len=16)
        mock_engine.assert_called_once_with(
            model_name="meta-llama/Llama-3.1-8B",
            batch_size=2,
            seq_len=16,
        )
        mock_judge.assert_called_once_with(fake_raw)
        assert res == fake_judged


def test_reverify_pass_verdict() -> None:
    fake_raw = {"status": "ok", "device": "cuda"}
    fake_judged = {"status": "ok", "device": "cuda", "verdict": "PASS", "reasons": []}

    with (
        patch("preflight.reverify.run_canary_check", return_value=fake_raw),
        patch("preflight.reverify.judge_result", return_value=fake_judged),
    ):
        res = reverify(None, 1, 8)
        assert res["verdict"] == "PASS"


def test_reverify_fail_verdict() -> None:
    fake_raw = {"status": "import_crash", "error_log": "CUDA error"}
    fake_judged = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["import_crash"],
        "error_log": "CUDA error",
    }

    with (
        patch("preflight.reverify.run_canary_check", return_value=fake_raw),
        patch("preflight.reverify.judge_result", return_value=fake_judged),
    ):
        res = reverify(None, 1, 8)
        assert res["verdict"] == "FAIL"
        assert res["reasons"] == ["import_crash"]


def test_cli_yes_calls_reverify_after_fix_success() -> None:
    fake_raw = {"status": "import_crash", "error_log": "CUDA Setup failed"}
    fake_initial = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["import_crash"],
        "error_log": "CUDA Setup failed",
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
        patch("preflight.cli.apply_fix") as mock_apply_fix,
        patch("preflight.cli.reverify", return_value=fake_reverified) as mock_reverify,
        patch("preflight.cli.render_report") as mock_render,
    ):
        result = runner.invoke(
            app, ["check", "--yes", "--model", "test-model", "--batch-size", "4"]
        )
        assert result.exit_code == 0
        mock_apply_fix.assert_called_once()
        mock_reverify.assert_called_once_with(
            model="test-model",
            batch_size=4,
            seq_len=8,
        )
        assert mock_render.call_count == 1
        rendered_data = mock_render.call_args[0][0][0]
        assert "reverified" in rendered_data
        assert rendered_data["reverified"] == fake_reverified


def test_cli_yes_exits_nonzero_when_reverify_fails() -> None:
    fake_raw = {"status": "import_crash", "error_log": "CUDA Setup failed"}
    fake_initial = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["import_crash"],
        "error_log": "CUDA Setup failed",
    }
    fake_reverified = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["import_crash"],
        "error_log": "CUDA Setup failed still",
    }

    with (
        patch("preflight.cli.run_canary_check", return_value=fake_raw),
        patch("preflight.cli.judge_result", return_value=fake_initial),
        patch("preflight.cli.apply_fix") as mock_apply_fix,
        patch("preflight.cli.reverify", return_value=fake_reverified) as mock_reverify,
        patch("preflight.cli.render_report") as mock_render,
    ):
        result = runner.invoke(app, ["check", "--yes"])
        assert result.exit_code == 1
        mock_apply_fix.assert_called_once()
        mock_reverify.assert_called_once()
        rendered_data = mock_render.call_args[0][0][0]
        assert rendered_data["reverified"]["verdict"] == "FAIL"


def test_cli_yes_not_called_when_initial_pass() -> None:
    fake_raw = {"status": "ok", "device": "cuda"}
    fake_initial = {
        "status": "ok",
        "device": "cuda",
        "verdict": "PASS",
        "reasons": [],
    }

    with (
        patch("preflight.cli.run_canary_check", return_value=fake_raw),
        patch("preflight.cli.judge_result", return_value=fake_initial),
        patch("preflight.cli.apply_fix") as mock_apply_fix,
        patch("preflight.cli.reverify") as mock_reverify,
        patch("preflight.cli.render_report"),
    ):
        result = runner.invoke(app, ["check", "--yes"])
        assert result.exit_code == 0
        mock_apply_fix.assert_not_called()
        mock_reverify.assert_not_called()


def test_cli_yes_not_called_when_apply_fix_raises_error() -> None:
    fake_raw = {"status": "import_crash", "error_log": "CUDA Setup failed"}
    fake_initial = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["import_crash"],
        "error_log": "CUDA Setup failed",
    }

    err = FixExecutionError(
        command="pip install foo",
        returncode=1,
        stdout="",
        stderr="pip fail",
    )

    with (
        patch("preflight.cli.run_canary_check", return_value=fake_raw),
        patch("preflight.cli.judge_result", return_value=fake_initial),
        patch("preflight.cli.apply_fix", side_effect=err) as mock_apply_fix,
        patch("preflight.cli.reverify") as mock_reverify,
        patch("preflight.cli.render_report") as mock_render,
    ):
        result = runner.invoke(app, ["check", "--yes"])
        assert result.exit_code != 0
        mock_apply_fix.assert_called_once()
        mock_reverify.assert_not_called()
        mock_render.assert_not_called()


def test_cli_without_yes_does_not_call_reverify() -> None:
    fake_raw = {"status": "import_crash", "error_log": "CUDA Setup failed"}
    fake_initial = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["import_crash"],
        "error_log": "CUDA Setup failed",
    }

    with (
        patch("preflight.cli.run_canary_check", return_value=fake_raw),
        patch("preflight.cli.judge_result", return_value=fake_initial),
        patch("preflight.cli.apply_fix") as mock_apply_fix,
        patch("preflight.cli.reverify") as mock_reverify,
        patch("preflight.cli.render_report"),
    ):
        result = runner.invoke(
            app,
            [
                "check",
            ],
        )
        assert result.exit_code == 1
        mock_apply_fix.assert_not_called()
        mock_reverify.assert_not_called()
