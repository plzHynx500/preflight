"""query_gpu_state() 테스트. 실제 pynvml/GPU가 없어도 로직을 검증한다 —
sys.modules에 가짜 pynvml을 심어 성공/실패 양쪽 경로를 확인한다.
"""

from __future__ import annotations

import sys
import types

import pytest


class _FakeMemInfo:
    def __init__(self, total_bytes: int, free_bytes: int) -> None:
        self.total = total_bytes
        self.free = free_bytes


@pytest.fixture
def fake_pynvml_ok(monkeypatch):
    """정상 조회되는 가짜 pynvml — 1개 GPU, 12GB 카드에 11.5GB 가용."""
    calls = {"shutdown": False}

    fake_module = types.SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: calls.__setitem__("shutdown", True),
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetName=lambda handle: b"NVIDIA GeForce RTX 4070 Ti",
        nvmlDeviceGetMemoryInfo=lambda handle: _FakeMemInfo(
            total_bytes=12 * 1024 * 1024 * 1024, free_bytes=int(11.5 * 1024 * 1024 * 1024)
        ),
        nvmlSystemGetDriverVersion=lambda: b"610.62",
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_module)
    return calls


def test_query_gpu_state_returns_expected_fields(fake_pynvml_ok) -> None:
    from preflight.gpu import query_gpu_state

    result = query_gpu_state()

    assert result is not None
    assert result["name"] == "NVIDIA GeForce RTX 4070 Ti"
    assert result["driver_version"] == "610.62"
    assert result["total_mb"] == pytest.approx(12 * 1024, rel=0.01)
    assert result["free_mb"] == pytest.approx(11.5 * 1024, rel=0.01)


def test_query_gpu_state_always_calls_shutdown(fake_pynvml_ok) -> None:
    """nvmlInit 성공 후에는 성공/실패와 무관하게 nvmlShutdown이 불려야 한다."""
    from preflight.gpu import query_gpu_state

    query_gpu_state()

    assert fake_pynvml_ok["shutdown"] is True


def test_query_gpu_state_returns_none_when_pynvml_not_installed(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "pynvml", raising=False)
    real_import = __import__

    def _blocking_import(name, *args, **kwargs):
        if name == "pynvml":
            raise ModuleNotFoundError("No module named 'pynvml'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _blocking_import)

    from preflight.gpu import query_gpu_state

    assert query_gpu_state() is None


def test_query_gpu_state_returns_none_when_nvml_init_fails(monkeypatch) -> None:
    """드라이버 미설치 등으로 nvmlInit() 자체가 실패하는 경우."""

    def _raise_init():
        raise RuntimeError("NVML Shared Library Not Found")

    fake_module = types.SimpleNamespace(nvmlInit=_raise_init)
    monkeypatch.setitem(sys.modules, "pynvml", fake_module)

    from preflight.gpu import query_gpu_state

    assert query_gpu_state() is None


def test_query_gpu_state_returns_none_when_query_fails_after_init(monkeypatch) -> None:
    """nvmlInit은 성공했지만 이후 조회가 실패하는 경우 — shutdown은 그래도 불린다."""
    calls = {"shutdown": False}

    def _raise_handle(index):
        raise RuntimeError("NVML: GPU가 감지되지 않음")

    fake_module = types.SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: calls.__setitem__("shutdown", True),
        nvmlDeviceGetHandleByIndex=_raise_handle,
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_module)

    from preflight.gpu import query_gpu_state

    assert query_gpu_state() is None
    assert calls["shutdown"] is True
