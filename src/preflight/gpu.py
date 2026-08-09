"""pynvml 기반 GPU/드라이버 상태 조회. docs/architecture.md §2 기술 스택 참고.

torch를 import하지 않는다 — nvidia-ml-py는 CUDA 컨텍스트를 만들지 않고
libnvidia-ml을 직접 호출하므로, 부모 프로세스(cli.py)에서 canary 기동 직전에
안전하게 호출할 수 있다. ADR-0002가 격리하려는 건 "import torch가 프로세스를
죽이는 것"이지 GPU 조회가 아니다(2026-08-06 회의 안건 3-1).
"""

from __future__ import annotations


def query_gpu_state(device_index: int = 0) -> dict | None:
    """GPU 이름·총/가용 VRAM·드라이버 버전을 조회한다.

    실패(NVML 미설치, 드라이버 없음, GPU 없음 등)하면 크래시 대신 None을
    돌려준다 — 호출 측(judge_result)이 이 값을 못 받으면 가용 VRAM 기반
    WARN 판정만 조용히 건너뛰고 나머지 판정은 정상 진행한다.

    **호출 시점이 중요하다** — canary 기동 "직전" 1회만 불러야 한다. canary가
    돌기 시작하면 free_mb가 canary 자신의 점유만큼 깎여 오염된 값이 된다.
    알고 싶은 건 "학습을 시작하려는 시점에 쓸 수 있는 양"이다.

    다중 GPU 환경은 MVP 범위 밖이라 `device_index=0`(기본 GPU)만 조회한다.
    """
    try:
        import pynvml
    except ImportError:
        return None

    try:
        pynvml.nvmlInit()
    except Exception:  # noqa: BLE001 - 드라이버 미설치 등 실패 종류와 무관하게 건너뛴다
        return None

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")

        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        driver_version = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver_version, bytes):
            driver_version = driver_version.decode("utf-8", errors="replace")

        return {
            "name": name,
            "total_mb": mem_info.total / (1024 * 1024),
            "free_mb": mem_info.free / (1024 * 1024),
            "driver_version": driver_version,
        }
    except Exception:  # noqa: BLE001 - 조회 실패 종류와 무관하게 None으로 우아하게 건너뛴다
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001, S110 - shutdown 실패는 호출자에게 새지 않는다
            pass
