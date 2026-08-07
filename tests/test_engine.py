"""CanaryEngine 테스트. 반환 스키마 계약은 docs/contracts/canary-api.md 참고.

스키마·예외 미발생 테스트는 자식 프로세스를 가짜로 바꿔 torch 없이도 돌아간다.
실제 canary 실행을 확인하는 테스트만 GPU를 요구한다.
"""

import json
import subprocess

import pytest

from preflight.canary import engine, worker
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


# ── 프로세스 격리 (W3 / #2) ──────────────────────────────────────────────────
#
# import 크래시를 흉내내는 대신 **진짜로 재현한다** — 가짜 `torch.py`를 임시
# 디렉터리에 만들고 PYTHONPATH 앞에 붙여 자식이 그걸 import하게 한다. 그러면
# 실제 사용자 환경에서 torch import가 깨지는 것과 같은 경로를 탄다.


@pytest.fixture
def fake_module(tmp_path, monkeypatch):
    """자식 프로세스가 import할 가짜 모듈을 심는다 (PYTHONPATH 앞에 붙인다)."""

    def _install(name: str, source: str) -> None:
        (tmp_path / f"{name}.py").write_text(source, encoding="utf-8")
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    return _install


def test_import_crash_is_normalized(fake_module) -> None:
    """`import torch`가 예외로 실패하면 status="import_crash"로 정규화된다."""
    fake_module("torch", 'raise ImportError("libcudart.so.12: cannot open shared object file")')

    result = run_canary_check(None, 1, 8)

    assert set(result) == set(RESULT_FIELDS)
    assert result["status"] == "import_crash"
    assert "libcudart" in result["error_log"]


@pytest.mark.parametrize(
    ("how", "source"),
    [
        # 널 포인터 역참조 — .so 로드 실패와 같은 유형의 진짜 네이티브 크래시
        ("segfault", "import ctypes\nctypes.string_at(0)\n"),
        # 파이썬 정리 절차를 전혀 안 거치는 즉시 종료
        ("os._exit", "import os\nos._exit(1)\n"),
    ],
)
def test_process_death_during_import_is_import_crash(fake_module, how, source) -> None:
    """예외조차 못 남기고 죽어도 import_crash로 잡힌다.

    이런 죽음에는 `except`도 `finally`도 돌지 않아서 "죽은 뒤 기록"이 불가능하다.
    자식이 **미리 써둔** 결과가 살아남아야 하고, 부모는 exit code를 해석하지 않는다
    (같은 크래시가 Linux는 -11, Windows는 0xC0000005로 나타난다 — NFR-01).
    """
    fake_module("torch", source)

    result = run_canary_check(None, 1, 8)

    assert set(result) == set(RESULT_FIELDS), how
    assert result["status"] == "import_crash", result["error_log"]
    assert result["error_log"]


@requires_cuda
def test_bitsandbytes_failure_falls_back_instead_of_crashing(fake_module) -> None:
    """bitsandbytes만 못 쓰는 환경은 크래시가 아니라 폴백이다.

    파이썬 예외로 잡히는 bnb 실패(미설치·구버전)는 진단을 멈출 이유가 아니므로
    `nn.Linear`로 대체해 계속 측정하고, 그 사실을 `quant_backend`로 알린다
    (docs/architecture.md §6-01). torch 자체가 죽는 경우와 구분돼야 한다.
    """
    fake_module("bitsandbytes", 'raise ImportError("no bitsandbytes here")')

    result = run_canary_check(None, 1, 8)

    assert result["status"] == "ok", result["error_log"]
    assert result["quant_backend"] == "nn-linear-fallback"
    assert result["device"] == "cuda"


def test_timeout_is_normalized(monkeypatch) -> None:
    """자식이 응답하지 않아도 예외 대신 정규화된 결과를 돌려준다."""

    def hang(cmd, **_kwargs):
        raise subprocess.TimeoutExpired(cmd, engine.WORKER_TIMEOUT_SEC)

    monkeypatch.setattr(engine.subprocess, "run", hang)

    result = run_canary_check(None, 1, 8)

    assert set(result) == set(RESULT_FIELDS)
    assert result["status"] == "error"
    assert "강제 종료" in result["error_log"]


@requires_cuda
def test_oom_is_normalized() -> None:
    """실제 CUDA OOM을 일으켜 status="oom"으로 정규화되는지 확인한다.

    메시지 문자열이 아니라 `torch.cuda.OutOfMemoryError` 타입으로 판별한다
    (docs/adr/0002-subprocess-isolation-for-canary.md).
    """
    # hidden=4096 기준 입력 텐서만 수십 GB — 어떤 소비자용 GPU에서도 확실히 넘친다.
    result = run_canary_check(None, 256, 8192)

    assert set(result) == set(RESULT_FIELDS)
    assert result["status"] == "oom", result["error_log"]
    assert result["error_log"]


@requires_cuda
def test_measure_retries_without_4bit_when_it_fails(monkeypatch) -> None:
    """4bit 구성이 실패하면 nn.Linear로 **한 번만** 더 시도한다.

    구버전 bitsandbytes가 CPU 4bit을 지원하지 않는 경우가 대표적이다
    (docs/architecture.md §6-01 "구버전 bnb 대응").
    """
    import torch

    attempts = []
    real_build = worker.build_minimal_canary_model

    def flaky(device, dtype, prefer_4bit=True):
        attempts.append(prefer_4bit)
        if prefer_4bit:
            raise RuntimeError("구버전 bitsandbytes: CPU 4bit 미지원")
        return real_build(device, dtype, prefer_4bit)

    monkeypatch.setattr(worker, "build_minimal_canary_model", flaky)

    measured = worker._measure(torch, "cpu", torch.float32, 1, 8, prefer_4bit=True)

    assert attempts == [True, False]
    assert measured["quant_backend"] == "nn-linear-fallback"
    assert measured["elapsed_ms"] > 0


@requires_cuda
def test_measure_does_not_retry_on_oom(monkeypatch) -> None:
    """OOM은 폴백으로 해결되지 않으므로 재시도하지 않고 그대로 올린다.

    nn.Linear로 다시 돌려도 메모리는 그대로 부족하다. 재시도하면 시간만 버리고
    원래 원인(oom)이 두 번째 실패에 가려진다.
    """
    import torch

    attempts = []

    def always_oom(device, dtype, prefer_4bit=True):
        attempts.append(prefer_4bit)
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(worker, "build_minimal_canary_model", always_oom)

    with pytest.raises(torch.cuda.OutOfMemoryError):
        worker._measure(torch, "cuda", torch.float16, 1, 8, prefer_4bit=True)

    assert attempts == [True]


def test_write_result_keeps_previous_record_on_failure(tmp_path, monkeypatch) -> None:
    """결과를 덮어쓰다 실패해도 직전 기록이 살아남는다 (원자적 쓰기).

    `open(path, "w")`는 여는 순간 기존 내용을 지우므로, 그 상태로 죽으면 미리 써둔
    정보까지 날아간다. 임시 파일에 쓴 뒤 rename하면 실패해도 원본이 무사하다.
    """
    result_path = str(tmp_path / "result.json")
    worker._write_result(result_path, {"status": "import_crash", "error_log": "미리 써둔 기록"})

    def boom(*_args, **_kwargs):
        raise OSError("디스크 가득 참")

    monkeypatch.setattr(worker.json, "dump", boom)
    with pytest.raises(OSError):
        worker._write_result(result_path, {"status": "ok"})

    with open(result_path, encoding="utf-8") as result_file:
        assert json.load(result_file)["status"] == "import_crash"


def test_write_result_leaves_no_temp_file(tmp_path) -> None:
    """성공적으로 쓰고 나면 임시 파일이 남지 않는다."""
    result_path = tmp_path / "result.json"
    worker._write_result(str(result_path), {"status": "ok"})

    assert result_path.exists()
    assert list(tmp_path.iterdir()) == [result_path]
