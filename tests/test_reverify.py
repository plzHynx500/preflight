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
        patch("preflight.reverify.query_gpu_state", return_value=None),
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


def test_reverify_merges_gpu_state_into_env() -> None:
    """재확인 경로에서도 가용 VRAM을 다시 조회해 env에 얹는다 (#68).

    이게 없으면 `judge_result`가 `gpu_free_mb` 부재를 이유로 `memory_delta_high`
    WARN을 조용히 건너뛴다 — VRAM이 빠듯한 환경의 WARN이 재확인에서는 구조적으로
    다시 나올 수 없고 늘 PASS로 뒤집힌다.
    """
    fake_raw = {"status": "ok", "device": "cuda", "env": {"bnb_compiled_with_cuda": True}}
    state = {"name": "RTX 4070 Ti", "total_mb": 12288.0, "free_mb": 512.0, "driver_version": "560"}

    with (
        patch("preflight.reverify.query_gpu_state", return_value=state) as mock_state,
        patch("preflight.reverify.run_canary_check", return_value=fake_raw),
        patch(
            "preflight.reverify.judge_result", side_effect=lambda raw: {**raw, "verdict": "WARN"}
        ),
    ):
        res = reverify(None, 1, 8)

    mock_state.assert_called_once()
    assert res["env"]["gpu_free_mb"] == 512.0
    assert res["env"]["gpu_total_mb"] == 12288.0
    # 자식이 채워 보낸 값은 그대로 남는다 — 부모는 뒤에 얹기만 한다.
    assert res["env"]["bnb_compiled_with_cuda"] is True


def test_reverify_pass_verdict() -> None:
    fake_raw = {"status": "ok", "device": "cuda"}
    fake_judged = {"status": "ok", "device": "cuda", "verdict": "PASS", "reasons": []}

    with (
        patch("preflight.reverify.query_gpu_state", return_value=None),
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
        patch("preflight.reverify.query_gpu_state", return_value=None),
        patch("preflight.reverify.run_canary_check", return_value=fake_raw),
        patch("preflight.reverify.judge_result", return_value=fake_judged),
    ):
        res = reverify(None, 1, 8)
        assert res["verdict"] == "FAIL"
        assert res["reasons"] == ["import_crash"]


def test_cli_yes_reverifies_the_check_that_produced_the_fix() -> None:
    """재확인 대상은 fix의 근거가 된 그 체크다 (#68).

    기본 체크가 FAIL이면 모델 체크는 fail-fast로 아예 돌지 않는다. 예전에는
    `--model`이 주어졌다는 이유만으로 그 한 번도 실행된 적 없는 모델 canary를
    재확인에서 돌렸고, 정작 고쳤는지 봐야 할 기본 체크는 다시 돌지 않았다.

    기본 체크 재확인이 PASS로 뒤집힌 뒤에는, fail-fast로 생략됐던 모델 체크를
    이어서 실행한다(#84) — 그러지 않으면 모델을 한 번도 확인한 적이 없는데
    exit code만 0이 되는, 이 테스트가 원래 검증하던 것과는 다른 새 버그가 된다.
    """
    fake_raw = {"status": "import_crash", "error_log": "CUDA Setup failed"}
    fake_initial = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["import_crash"],
        "error_log": "libbitsandbytes_cpu.so: CUDA Setup failed",
    }
    fake_reverified = {
        "status": "ok",
        "device": "cuda",
        "verdict": "PASS",
        "reasons": [],
    }
    fake_raw_model = {"status": "ok"}
    fake_model_res = {"status": "ok", "device": "cuda", "verdict": "PASS", "reasons": []}

    with (
        patch("preflight.cli.query_gpu_state", return_value=None),
        patch(
            "preflight.cli.run_canary_check", side_effect=[fake_raw, fake_raw_model]
        ) as mock_run,
        patch(
            "preflight.cli.judge_result", side_effect=[fake_initial, fake_model_res]
        ),
        patch("preflight.cli.apply_fix") as mock_apply_fix,
        patch("preflight.cli.reverify", return_value=fake_reverified) as mock_reverify,
        patch("preflight.cli.render_report") as mock_render,
    ):
        result = runner.invoke(
            app, ["check", "--yes", "--model", "test-model", "--batch-size", "4"]
        )
        assert result.exit_code == 0
        mock_apply_fix.assert_called_once()
        # 기본 체크(batch=1, seq=8 고정)가 FAIL이었으므로 그 조건으로 재실행한다.
        mock_reverify.assert_called_once_with(
            model_name=None,
            batch_size=1,
            seq_len=8,
        )
        # 기본 체크(초기 1회) + 생략됐던 모델 체크(재확인 후 이어서 1회) = 2회.
        assert mock_run.call_count == 2
        assert mock_render.call_count == 1
        results = mock_render.call_args[0][0]
        assert results[0]["reverified"] is True
        assert results[0]["verdict"] == "PASS"
        assert results[0]["status"] == "ok"
        assert "skipped" not in results[1]
        assert results[1]["verdict"] == "PASS"
        assert results[1]["model_name"] == "test-model"


def test_cli_yes_reverifies_model_check_with_its_own_size() -> None:
    """모델 체크가 fix 대상이면 --batch-size·--seq-len을 그대로 물려준다 (#68)."""
    basic = {"status": "ok", "device": "cuda", "verdict": "PASS", "reasons": []}
    model_fail = {
        "status": "oom",
        "device": "cuda",
        "verdict": "FAIL",
        "reasons": ["status_oom"],
    }
    reverified = {"status": "ok", "device": "cuda", "verdict": "PASS", "reasons": []}

    with (
        patch("preflight.cli.run_canary_check", return_value={"status": "ok"}),
        patch("preflight.cli.judge_result", side_effect=[basic, model_fail]),
        patch("preflight.cli.apply_fix"),
        patch("preflight.cli.suggest_fix", return_value={"cause": "oom", "fix_argv": ["x"]}),
        patch("preflight.cli.reverify", return_value=reverified) as mock_reverify,
        patch("preflight.cli.render_report"),
    ):
        result = runner.invoke(
            app,
            ["check", "--yes", "--model", "test-model", "--batch-size", "4", "--seq-len", "2048"],
        )

    assert result.exit_code == 0
    mock_reverify.assert_called_once_with(
        model_name="test-model",
        batch_size=4,
        seq_len=2048,
    )


def test_cli_yes_keeps_verdict_of_results_not_reverified() -> None:
    """재확인 1건이 전체 verdict를 대체하지 않는다 (#68).

    기본 체크 WARN + 모델 체크 FAIL에서 모델만 재확인해 PASS가 되어도, 기본
    체크의 WARN은 그대로 남아야 한다 — 예전에는 종료 코드가 0이 되어 WARN이
    통째로 사라졌다.
    """
    basic_warn = {
        "status": "ok",
        "device": "cuda",
        "verdict": "WARN",
        "reasons": ["cpu_multiplier_low"],
    }
    model_fail = {"status": "oom", "device": "cuda", "verdict": "FAIL", "reasons": ["status_oom"]}
    reverified = {"status": "ok", "device": "cuda", "verdict": "PASS", "reasons": []}

    with (
        patch("preflight.cli.run_canary_check", return_value={"status": "ok"}),
        patch("preflight.cli.judge_result", side_effect=[basic_warn, model_fail]),
        patch("preflight.cli.apply_fix"),
        patch("preflight.cli.suggest_fix", return_value={"cause": "oom", "fix_argv": ["x"]}),
        patch("preflight.cli.reverify", return_value=reverified),
        patch("preflight.cli.render_report"),
    ):
        result = runner.invoke(app, ["check", "--yes", "--model", "test-model"])

    assert result.exit_code == 2


def test_cli_yes_exits_nonzero_when_reverify_fails() -> None:
    fake_raw = {"status": "import_crash", "error_log": "libbitsandbytes_cpu.so: CUDA Setup failed"}
    fake_initial = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["import_crash"],
        "error_log": "libbitsandbytes_cpu.so: CUDA Setup failed",
    }
    fake_reverified = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["import_crash"],
        "error_log": "libbitsandbytes_cpu.so: CUDA Setup failed still",
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
        assert rendered_data["reverified"] is True
        assert rendered_data["verdict"] == "FAIL"
        assert rendered_data["error_log"] == "libbitsandbytes_cpu.so: CUDA Setup failed still"


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


def test_cli_yes_reports_fix_failure_without_traceback() -> None:
    """apply_fix가 실패해도 트레이스백이 새지 않고 안내로 바뀐다 (#53).

    예전에는 `FixExecutionError`가 그대로 위로 올라가 날것의 파이썬 트레이스백이
    화면에 찍혔다 — 에러를 읽기 쉽게 만들어주는 것이 목적인 도구의 가장 나쁜
    실패 방식이다. 진단 결과는 그대로 보여주고, 재확인만 건너뛴다.
    """
    fake_raw = {"status": "import_crash", "error_log": "CUDA Setup failed"}
    fake_initial = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["import_crash"],
        "error_log": "libbitsandbytes_cpu.so: CUDA Setup failed",
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

    # 1차 판정이 FAIL이었고 재확인을 한 적이 없으므로 1차 판정 기준 종료 코드다.
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "Traceback" not in result.output
    mock_apply_fix.assert_called_once()
    mock_reverify.assert_not_called()
    # 진단 결과는 그대로 보여준다 — 수정이 실패했다고 진단까지 감출 이유는 없다.
    mock_render.assert_called_once()
    notices = mock_render.call_args.kwargs["notices"]
    assert any("자동 수정 실패" in n for n in notices)
    assert any("종료 코드 1" in n for n in notices)


def test_cli_yes_fix_failure_keeps_first_pass_warn_exit_code() -> None:
    """수정 실패 시 종료 코드는 1차 판정 기준이다 (#53) — WARN이면 2다."""
    fake_initial = {
        "status": "ok",
        "device": "cuda",
        "verdict": "WARN",
        "reasons": ["cpu_multiplier_low"],
    }
    err = FixExecutionError(command="pip install foo", returncode=None, stdout="", stderr="no exe")

    with (
        patch("preflight.cli.run_canary_check", return_value={"status": "ok"}),
        patch("preflight.cli.judge_result", return_value=fake_initial),
        patch(
            "preflight.cli.suggest_fix",
            return_value={"cause": "cpu_multiplier_low", "fix_argv": ["x"], "fix_command": "x"},
        ),
        patch("preflight.cli.apply_fix", side_effect=err),
        patch("preflight.cli.reverify") as mock_reverify,
        patch("preflight.cli.render_report") as mock_render,
    ):
        result = runner.invoke(app, ["check", "--yes"])

    assert result.exit_code == 2
    mock_reverify.assert_not_called()
    notices = mock_render.call_args.kwargs["notices"]
    assert any("실행조차 하지 못했다" in n for n in notices)


def test_cli_yes_skips_reverify_when_no_fix_command() -> None:
    """실행할 명령이 없으면 canary를 다시 돌리지 않고 그 사실을 알린다 (#57).

    `suggest_fix`는 `fix_command`가 None이어도 dict를 돌려주므로 예전 조건
    (`if yes and fix:`)은 항상 참이었다. 그래서 아무것도 실행하지 않은 채
    canary만 한 번 더 돌아(실측 11초 → 17초) 화면은 그대로였다.
    """
    fake_initial = {
        "status": "ok",
        "device": "cpu",
        "quant_backend": "bnb-4bit",
        "verdict": "FAIL",
        "reasons": ["quant_layer_device_cpu"],
        "env": {"bnb_compiled_with_cuda": True},
    }

    with (
        patch("preflight.cli.run_canary_check", return_value={"status": "ok"}),
        patch("preflight.cli.judge_result", return_value=fake_initial),
        patch("preflight.cli.apply_fix") as mock_apply_fix,
        patch("preflight.cli.reverify") as mock_reverify,
        patch("preflight.cli.render_report") as mock_render,
    ):
        result = runner.invoke(app, ["check", "--yes"])

    assert result.exit_code == 1
    mock_apply_fix.assert_not_called()
    mock_reverify.assert_not_called()
    notices = mock_render.call_args.kwargs["notices"]
    assert any("자동으로 실행할 수정 명령이 없어" in n for n in notices)


def test_cli_without_yes_does_not_call_reverify() -> None:
    fake_raw = {"status": "import_crash", "error_log": "libbitsandbytes_cpu.so: CUDA Setup failed"}
    fake_initial = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["import_crash"],
        "error_log": "libbitsandbytes_cpu.so: CUDA Setup failed",
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
