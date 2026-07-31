"""CanaryEngine 테스트. 반환 스키마 계약은 docs/contracts/canary-api.md 참고.

스키마·예외 미발생 테스트는 자식 프로세스를 가짜로 바꿔 torch 없이도 돌아간다.
실제 canary 실행을 확인하는 테스트만 GPU를 요구한다.
"""

import json
import subprocess

from preflight.canary import engine
from preflight.canary.engine import RESULT_FIELDS, run_canary_check
from tests.conftest import requires_cuda

_OK_PAYLOAD = {
    "status": "ok",
    "device": "cuda",
    "memory_delta_mb": 42.0,
    "elapsed_ms": 12.3,
    "cpu_multiplier": 41.0,
    "quant_backend": "bnb-4bit",
    "error_log": None,
}


def _fake_worker(payload, returncode=0, stderr=""):
    """자식 프로세스를 대신해 결과 파일을 쓰는 subprocess.run 대역."""

    def fake_run(cmd, **_kwargs):
        if payload is not None:
            with open(cmd[-1], "w", encoding="utf-8") as result_file:
                json.dump(payload, result_file, ensure_ascii=False)
        return subprocess.CompletedProcess(cmd, returncode, "", stderr)

    return fake_run


def test_run_canary_check_returns_normalized_schema(monkeypatch) -> None:
    monkeypatch.setattr(engine.subprocess, "run", _fake_worker(_OK_PAYLOAD))

    result = run_canary_check(None, 1, 8)

    assert set(result) == set(RESULT_FIELDS)
    assert result == _OK_PAYLOAD


def test_run_canary_check_drops_unknown_fields_and_fills_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        engine.subprocess, "run", _fake_worker({"status": "ok", "device": "cuda", "extra": 1})
    )

    result = run_canary_check(None, 1, 8)

    assert set(result) == set(RESULT_FIELDS)
    assert result["memory_delta_mb"] is None


def test_run_canary_check_normalizes_dead_child(monkeypatch) -> None:
    """자식이 결과를 남기지 못하고 죽어도 예외 대신 정규화된 dict를 돌려준다."""
    monkeypatch.setattr(
        engine.subprocess, "run", _fake_worker(None, returncode=-11, stderr="Segmentation fault")
    )

    result = run_canary_check(None, 1, 8)

    assert set(result) == set(RESULT_FIELDS)
    assert result["status"] == "error"
    assert "Segmentation fault" in result["error_log"]


def test_run_canary_check_rejects_unknown_status(monkeypatch) -> None:
    monkeypatch.setattr(engine.subprocess, "run", _fake_worker({"status": "weird"}))

    result = run_canary_check(None, 1, 8)

    assert result["status"] == "error"
    assert "weird" in result["error_log"]


def test_run_canary_check_survives_subprocess_failure(monkeypatch) -> None:
    """subprocess 기동 자체가 실패해도 예외가 호출자에게 새어나가지 않는다."""

    def explode(*_args, **_kwargs):
        raise OSError("subprocess를 띄울 수 없음")

    monkeypatch.setattr(engine.subprocess, "run", explode)

    result = run_canary_check(None, 1, 8)

    assert result["status"] == "error"
    assert "subprocess를 띄울 수 없음" in result["error_log"]


def test_run_canary_check_model_path_is_not_wired_yet() -> None:
    """`--model` 경로는 W4(이인수)·W9 이후에 연결된다 — 그때까지도 죽지는 않아야 한다.

    실제 자식 프로세스를 띄우는 유일한 비-GPU 테스트다. torch가 없는 환경에서는
    import 실패로, 있는 환경에서는 build_dummy_model()의 NotImplementedError로
    끝나지만, 어느 쪽이든 부모는 정규화된 결과를 받는다.
    """
    result = run_canary_check("some/model", 1, 8)

    assert set(result) == set(RESULT_FIELDS)
    assert result["status"] == "error"
    assert result["error_log"]


@requires_cuda
def test_basic_check_runs_on_gpu() -> None:
    """실제 canary를 GPU에서 돌려 device·메모리·시간이 측정되는지 확인한다."""
    result = run_canary_check(None, 1, 8)

    assert result["status"] == "ok", result["error_log"]
    assert result["device"] == "cuda"
    assert result["memory_delta_mb"] > 0
    assert result["elapsed_ms"] > 0
    assert result["cpu_multiplier"] is None or result["cpu_multiplier"] > 0
    assert result["quant_backend"] in ("bnb-4bit", "nn-linear-fallback")
