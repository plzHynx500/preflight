from __future__ import annotations

import subprocess
import sys
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from preflight.canary.worker import _PREWRITE_IMPORT_NOTE
from preflight.cli import app
from preflight.fix.causes import classify_cause
from preflight.fix.executor import FixExecutionError, apply_fix, suggest_fix

BNB_REINSTALL_ARGS = ["-m", "pip", "install", "bitsandbytes", "--upgrade", "--force-reinstall"]


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
        "reasons": ["status_import_crash"],
        "error_log": "libbitsandbytes_cpu.so CUDA Setup failed",
    }
    assert classify_cause(res1) == "bnb_not_compiled_with_cuda"
    fix1 = suggest_fix(res1)
    assert fix1 is not None
    assert fix1["cause"] == "bnb_not_compiled_with_cuda"
    assert fix1["fix_argv"] == [sys.executable, *BNB_REINSTALL_ARGS]

    # case 2: 4bit cpu fallback — canary 자식이 채워 보낸 env로 판별한다 (#19).
    # torch 쪽 신호(torch_version/torch_cuda_version)를 아예 못 읽은 경우에만 남는
    # 마지막 폴백 경로다 — bnb_compiled_with_cuda 단독 판단의 한계는 #72 참고.
    res2 = {
        "status": "ok",
        "device": "cpu",
        "quant_backend": "bnb-4bit",
        "verdict": "FAIL",
        "reasons": ["quant_layer_device_cpu"],
        "env": {"bnb_compiled_with_cuda": False},
    }
    assert classify_cause(res2) == "bnb_not_compiled_with_cuda"
    fix2 = suggest_fix(res2)
    assert fix2 is not None
    assert fix2["cause"] == "bnb_not_compiled_with_cuda"
    assert fix2["fix_argv"] == [sys.executable, *BNB_REINSTALL_ARGS]


def test_suggest_fix_targets_the_running_python(monkeypatch) -> None:
    """수정은 **지금 preflight를 돌리고 있는 파이썬**에 한다 (#52).

    PATH에서 찾은 `pip`을 쓰면 venv를 활성화하지 않았거나 pipx/전역 설치로
    쓰는 사람에게 "진단은 A 환경, 수정은 B 환경"이 되고, 재확인이 계속 실패해도
    사용자는 이유를 알 수 없다.
    """
    monkeypatch.setattr(sys, "executable", r"C:\Program Files\Py 3.11\python.exe")

    fix = suggest_fix(
        {
            "status": "import_crash",
            "verdict": "FAIL",
            "reasons": ["status_import_crash"],
            "error_log": "libbitsandbytes_cpu.so CUDA Setup failed",
        }
    )

    assert fix is not None
    assert fix["fix_argv"] == [r"C:\Program Files\Py 3.11\python.exe", *BNB_REINSTALL_ARGS]
    # 화면에 찍히는 문자열도 그대로 복사해 쓸 수 있어야 한다 — 공백 있는 경로는
    # 큰따옴표로 감싼다(cmd·PowerShell·bash 모두에서 통하는 유일한 인용).
    assert fix["fix_command"] == (
        r'"C:\Program Files\Py 3.11\python.exe"'
        " -m pip install bitsandbytes --upgrade --force-reinstall"
    )
    # 짧게 보이려고 "python -m pip"으로 줄이지 않는다 — 복사해 붙이는 순간
    # 다시 "지금 활성화된 파이썬"으로 돌아가 같은 버그가 재현된다.
    assert not fix["fix_command"].startswith("pip ")


def test_classify_and_suggest_import_general() -> None:
    res = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["status_import_crash"],
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
        "device": "cpu",
        "quant_backend": "bnb-4bit",
        "verdict": "FAIL",
        "reasons": ["quant_layer_device_cpu"],
        "env": {"bnb_compiled_with_cuda": True},
    }
    assert classify_cause(res) == "4bit_cpu_fallback_other"
    fix = suggest_fix(res)
    assert fix is not None
    assert fix["cause"] == "4bit_cpu_fallback_other"
    assert fix["fix_command"] is None
    # 원인을 특정 못 한 경우다 — "CUDA 메모리 부족"이라고 단정하면 사용자를 정반대
    # 방향(배치 축소)으로 보낸다(#55).
    assert "메모리 부족" not in fix["message"]


# ── import_crash 분류: bitsandbytes 시그니처로 좁힌다 (#51) ────────────────────
#
# 로그에 "cuda"라는 글자가 있기만 하면 bitsandbytes 탓으로 돌리던 규칙 때문에,
# torch CUDA 빌드가 깨졌거나 CUDA 런타임이 없는 환경까지 전부 bnb 재설치로
# 안내됐다. bnb_not_compiled_with_cuda는 fix_command를 가진 유일한 원인이라
# 이 과잉 분류가 곧 "--yes가 무관한 명령을 실제로 실행"으로 직결됐다.

_CUDA_IMPORT_CRASH_LOGS = {
    "torch CUDA 빌드 깨짐": (
        "Traceback (most recent call last):\n"
        '  File "site-packages/torch/__init__.py", line 148, in <module>\n'
        "    raise err\n"
        "OSError: [WinError 126] 지정된 모듈을 찾을 수 없습니다. Error loading "
        '"C:\\venv\\Lib\\site-packages\\torch\\lib\\c10_cuda.dll" or one of its dependencies.'
    ),
    "CUDA 런타임 없음": (
        "Traceback (most recent call last):\n"
        '  File "site-packages/torch/__init__.py", line 1477, in <module>\n'
        "    from torch._C import *\n"
        "ImportError: libcudart.so.12: cannot open shared object file: "
        "No such file or directory"
    ),
}


@pytest.mark.parametrize("label", sorted(_CUDA_IMPORT_CRASH_LOGS))
def test_classify_cuda_import_crash_is_not_blamed_on_bitsandbytes(label: str) -> None:
    """로그에 cuda가 있다는 이유만으로 bitsandbytes 재설치를 안내하지 않는다(#51)."""
    res = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["status_import_crash"],
        "error_log": _CUDA_IMPORT_CRASH_LOGS[label],
    }

    assert classify_cause(res) == "import_crash_general"
    fix = suggest_fix(res)
    assert fix is not None
    assert fix["fix_command"] is None, "무관한 명령이 --yes로 실행되면 안 된다"


def test_classify_real_bitsandbytes_import_crash_still_matches() -> None:
    """진짜 bitsandbytes 로그는 계속 올바르게 분류된다 — 좁히다가 놓치면 안 된다(#51)."""
    logs = [
        "libbitsandbytes_cpu.so: undefined symbol: cget_col_row_stats",
        (
            "Traceback (most recent call last):\n"
            '  File "site-packages/bitsandbytes/cextension.py", line 109, in <module>\n'
            "    lib = get_native_library()\n"
            "RuntimeError: CUDA Setup failed despite CUDA being available."
        ),
    ]
    for log in logs:
        res = {
            "status": "import_crash",
            "verdict": "FAIL",
            "reasons": ["status_import_crash"],
            "error_log": log,
        }
        assert classify_cause(res) == "bnb_not_compiled_with_cuda", log


# ── 라이브러리 미설치는 설치 여부(find_spec)로 가른다 (#44, #92) ──────────────


def _import_crash(env: dict, error_log: str = "ModuleNotFoundError: No module named 'x'") -> dict:
    """import_crash 판정 결과에 주어진 env를 얹는다."""
    return {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["status_import_crash"],
        "error_log": error_log,
        "env": env,
    }


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"torch_installed": False}, "torch_not_installed_no_gpu"),
        (
            {"torch_installed": False, "gpu_free_mb": 9000.0},
            "torch_not_installed",
        ),
        ({"torch_installed": True, "transformers_installed": False}, "transformers_not_installed"),
    ],
)
def test_missing_library_is_classified_from_installed_flag(env: dict, expected: str) -> None:
    """설치 여부는 로그 문자열이 아니라 `env.*_installed`로 판별한다 (#44, #92).

    트레이스백 문자열을 뒤지면 문구가 바뀔 때마다 흔들리고, 네이티브 즉사처럼
    트레이스백이 아예 없는 경우엔 쓸 수도 없다.
    """
    assert classify_cause(_import_crash(env)) == expected


@pytest.mark.parametrize(
    ("driver", "expected_tag"),
    [("610.62", "cu130"), ("560.76", "cu126"), (None, "cu124")],
)
def test_torch_not_installed_gets_driver_matched_wheel(driver, expected_tag: str) -> None:
    """torch가 아예 없어도 드라이버에 맞는 CUDA 휠을 제안한다 (#44).

    `torch_cpu_only_build`와 같은 기계를 쓴다(#82, ADR-0007) — 맨몸
    `pip install torch`는 CPU 전용 빌드를 깔아 "고쳤는데 여전히 CPU"가 되므로,
    `--index-url`로 휠을 지정해야 한다.
    """
    result = _import_crash(
        {"torch_installed": False, "gpu_free_mb": 9000.0, "gpu_driver_version": driver}
    )

    fix = suggest_fix(result)
    assert fix is not None
    assert f"/whl/{expected_tag}" in fix["fix_command"]
    assert fix["fix_argv"][0] == sys.executable


@pytest.mark.parametrize(
    "cause_env",
    [
        {"torch_installed": False},  # GPU가 안 보임 -> torch_not_installed_no_gpu
        {"torch_installed": True, "transformers_installed": False},
    ],
)
def test_no_auto_fix_when_command_cannot_be_decided(cause_env: dict) -> None:
    """어떤 휠을 깔지 단정할 수 없으면 명령을 주지 않는다.

    GPU가 안 보이는 기계에 2GB CUDA 휠을 받게 하거나, torch 버전과 호환이 얽힌
    transformers를 자동 설치하면 기존 환경을 흔든다 (ADR-0008).
    """
    fix = suggest_fix(_import_crash(cause_env))

    assert fix is not None


# ── 사전 기록 문구는 원인 분류 시그니처와 겹치면 안 된다 (#93) ─────────────────


def test_prewritten_import_crash_does_not_name_a_library() -> None:
    """네이티브 즉사 결과가 특정 라이브러리 탓으로 분류되지 않는다 (#93).

    `.so` 로드 실패로 프로세스가 즉사하면 자식은 아무것도 쓰지 못하고, 부모는
    worker가 **미리 써둔 문구**를 그대로 `error_log`로 읽는다. 그 문구에
    `classify_cause`의 시그니처 문자열(`bitsandbytes` 등)이 들어 있으면
    **무엇이 죽었든 그 라이브러리 탓**이 된다 — torch의 .so가 죽어도
    "bitsandbytes를 재설치하라"가 나가고, 그 원인은 `fix_command`를 가진
    몇 안 되는 원인이라 `--yes`면 무관한 재설치가 실제로 실행된다.

    문구를 그대로 참조한다(복사하지 않는다) — 복사하면 나중에 worker의 문구를
    바꿔도 이 테스트가 안 깨져서 회귀를 못 잡는다.
    """
    result = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["status_import_crash"],
        "error_log": _PREWRITE_IMPORT_NOTE,
        "env": {},
    }

    assert classify_cause(result) == "import_crash_general"

    fix = suggest_fix(result)
    assert fix is not None
    # 원인을 특정하지 못한 상태이므로 --yes가 실행할 명령이 없어야 한다.
    assert fix["fix_command"] is None
    assert fix["fix_argv"] is None


def test_torch_is_checked_before_transformers() -> None:
    """둘 다 없으면 torch를 먼저 짚는다 — 없으면 모델 체크까지 갈 일이 없다."""
    result = _import_crash(
        {"torch_installed": False, "transformers_installed": False, "gpu_free_mb": 9000.0}
    )

    assert classify_cause(result) == "torch_not_installed"


@pytest.mark.parametrize("env", [{}, {"torch_installed": None}])
def test_unknown_installed_flag_does_not_claim_missing(env: dict) -> None:
    """`None`(못 읽음)을 "설치 안 됨"으로 단정하지 않는다 (#44).

    `find_spec` 자체가 실패할 수 있다. 그때는 기존 분류로 흘려보낸다 — "모르는 것"과
    "아닌 것"을 섞으면 멀쩡한 환경에 엉뚱한 안내가 나간다.
    """
    assert classify_cause(_import_crash(env)) == "import_crash_general"


def test_installed_flag_does_not_shadow_real_bitsandbytes_crash() -> None:
    """설치는 돼 있는데 로드가 깨진 경우는 기존 분류가 그대로 살아 있다."""
    result = _import_crash(
        {"torch_installed": True, "transformers_installed": True, "bitsandbytes_installed": True},
        error_log="libbitsandbytes_cpu.so: undefined symbol",
    )

    assert classify_cause(result) == "bnb_not_compiled_with_cuda"


# ── cpu 폴백 분류: env의 환경 사실을 읽는다 (#55, #72) ─────────────────────────


def _cpu_fallback(env: dict) -> dict:
    """4bit 레이어가 cpu로 떨어진 판정 결과에 주어진 env를 얹는다."""
    return {
        "status": "ok",
        "device": "cpu",
        "quant_backend": "bnb-4bit",
        "verdict": "FAIL",
        "reasons": ["quant_layer_device_cpu"],
        "env": env,
    }


def test_classify_cpu_only_torch_build_with_gpu_present() -> None:
    """torch가 CPU 전용 빌드인데 NVIDIA GPU는 보인다 → torch 재설치가 실제 해결책(#55)."""
    res = _cpu_fallback(
        {
            "torch_version": "2.13.0+cpu",
            "torch_cuda_version": None,
            "bnb_compiled_with_cuda": None,
            "gpu_free_mb": 9595.8,
            "gpu_total_mb": 12282.0,
        }
    )

    assert classify_cause(res) == "torch_cpu_only_build"
    fix = suggest_fix(res)
    assert fix is not None
    assert "CPU 전용 빌드" in fix["message"]
    assert fix["fix_command"] is not None
    assert "torch" in fix["fix_command"]
    # --force-reinstall이 없으면 pip이 "이미 설치됨"으로 아무것도 하지 않는다.
    assert "--force-reinstall" in fix["fix_command"]
    # env에 gpu_driver_version이 없으면 기존 기본값(cu124)으로 떨어진다.
    assert "/whl/cu124" in fix["fix_command"]


@pytest.mark.parametrize(
    ("driver_version", "expected_tag"),
    [
        ("595.79", "cu130"),  # major 595 >= 580 → CUDA 13.0 GA 구간
        ("580.65.06", "cu130"),  # 경계값 그대로
        ("572.13", "cu126"),  # major 572 >= 560이지만 580 미만
        ("560.76", "cu126"),  # 경계값 그대로 (Windows 12.6 최소)
        ("551.61", "cu124"),  # major 551 < 560 → 매핑 밖, 기본값
        ("560", "cu126"),  # 소수점 없이 major만 온 경우
        ("", "cu124"),  # 빈 문자열
        ("unknown", "cu124"),  # 숫자가 아닌 값
    ],
)
def test_torch_cpu_only_build_picks_cuda_wheel_by_driver_major(
    driver_version: str, expected_tag: str
) -> None:
    """torch 재설치 fix_command의 CUDA 휠 태그는 드라이버 major 브랜치 번호로 정해진다

    (#82, ADR-0007). patch 단위는 보지 않는다 — Linux/Windows 최소 드라이버 표의
    patch 값이 서로 달라 OS 판별 없이는 못 맞춘다.
    """
    res = _cpu_fallback(
        {
            "torch_version": "2.13.0+cpu",
            "torch_cuda_version": None,
            "bnb_compiled_with_cuda": None,
            "gpu_free_mb": 9595.8,
            "gpu_total_mb": 12282.0,
            "gpu_driver_version": driver_version,
        }
    )

    assert classify_cause(res) == "torch_cpu_only_build"
    fix = suggest_fix(res)
    assert fix is not None
    assert fix["fix_command"] is not None
    assert f"/whl/{expected_tag}" in fix["fix_command"]
    assert fix["fix_argv"][-1] == f"https://download.pytorch.org/whl/{expected_tag}"


@pytest.mark.parametrize(
    ("gpu_name", "driver_version", "expected_tag"),
    [
        # 드라이버 major가 cu126 구간이어도 Blackwell GeForce면 cu128로 끌어올린다
        # — cu126(CUDA 12.6)에는 sm_120 커널이 아예 없다(#102).
        ("NVIDIA GeForce RTX 5070 Laptop GPU", "572.13", "cu128"),
        ("NVIDIA GeForce RTX 5090", "560.76", "cu128"),
        # 기본값(매핑 밖) 구간도 마찬가지로 끌어올린다.
        ("NVIDIA GeForce RTX 5060 Ti", "551.61", "cu128"),
        # 드라이버가 이미 cu130 구간이면 더 최신이라 그대로 둔다 (override 불필요).
        ("NVIDIA GeForce RTX 5070 Laptop GPU", "595.79", "cu130"),
        # Blackwell이 아닌 GPU는 기존 드라이버 기반 매핑 그대로.
        ("NVIDIA GeForce RTX 4070 Ti", "572.13", "cu126"),
        # 이름이 없거나 조회 실패해도 기존 동작 그대로.
        (None, "572.13", "cu126"),
    ],
)
def test_torch_cpu_only_build_blackwell_geforce_forces_cu128(
    gpu_name: str | None, driver_version: str, expected_tag: str
) -> None:
    """Blackwell(RTX 50 시리즈) GPU는 드라이버 major 매핑 결과와 무관하게 최소

    cu128을 보장한다 (#102, ADR-0009) — cu124/cu126 빌드는 sm_120 커널이 없어
    드라이버를 올려도 'no kernel image is available' 오류가 그대로 재현된다.
    """
    res = _cpu_fallback(
        {
            "torch_version": "2.13.0+cpu",
            "torch_cuda_version": None,
            "bnb_compiled_with_cuda": None,
            "gpu_free_mb": 9595.8,
            "gpu_total_mb": 12282.0,
            "gpu_driver_version": driver_version,
            "gpu_name": gpu_name,
        }
    )

    fix = suggest_fix(res)
    assert fix is not None
    assert f"/whl/{expected_tag}" in fix["fix_command"]
    assert fix["fix_argv"][-1] == f"https://download.pytorch.org/whl/{expected_tag}"


def test_classify_cpu_only_torch_build_without_nvidia_gpu() -> None:
    """#55의 QA 환경(CPU 전용 torch + NVIDIA GPU 없음) — 원인은 CPU 빌드,
    다만 GPU가 없으니 CUDA torch를 자동으로 받아봐야 달라지는 게 없다.

    안내 문구에는 재설치 명령이 남아 "GPU 기계라면 이걸 하면 된다"를 알려주되,
    `--yes`의 자동 실행 대상에서는 뺀다.
    """
    res = _cpu_fallback(
        {
            "torch_version": "2.13.0+cpu",
            "torch_cuda_version": None,
            "bnb_compiled_with_cuda": None,
            "bnb_cpu_4bit_supported": None,
        }
    )

    assert classify_cause(res) == "torch_cpu_only_build_no_gpu"
    fix = suggest_fix(res)
    assert fix is not None
    assert "CPU 전용 빌드" in fix["message"]
    assert "torch" in fix["message"] and "재설치" in fix["message"]
    assert fix["fix_command"] is None
    assert "메모리 부족" not in fix["message"]


def test_classify_cuda_build_without_nvml_gpu() -> None:
    """torch는 CUDA 빌드인데 NVML이 GPU를 못 찾았다 → 드라이버/GPU 쪽 문제(#55)."""
    res = _cpu_fallback(
        {
            "torch_version": "2.11.0+cu128",
            "torch_cuda_version": "12.8",
            "bnb_compiled_with_cuda": False,
        }
    )

    assert classify_cause(res) == "no_nvidia_gpu_or_driver"
    fix = suggest_fix(res)
    assert fix is not None
    assert fix["fix_command"] is None
    assert "bitsandbytes" not in fix["message"]


def test_classify_hidden_cuda_device_is_not_a_bitsandbytes_build_problem() -> None:
    """#72 실측: CUDA_VISIBLE_DEVICES=-1이면 bnb_compiled_with_cuda가 False가 된다.

    bitsandbytes 0.50에서 이 값은 빌드 속성이 아니라 "런타임에 CUDA 장치가 보이는가"라
    GPU가 가려졌을 뿐인 정상 설치본에도 False가 나온다. 그걸 빌드 문제로 읽고
    재설치를 권하면 아무것도 안 고쳐진다 (RTX 4070 Ti / bnb 0.50.0 실측).
    """
    res = _cpu_fallback(
        {
            "torch_version": "2.11.0+cu128",
            "torch_cuda_version": "12.8",
            "bnb_compiled_with_cuda": False,
            "gpu_free_mb": 9595.8,
            "gpu_total_mb": 12282.0,
        }
    )

    assert classify_cause(res) == "cuda_device_not_visible"
    fix = suggest_fix(res)
    assert fix is not None
    assert fix["cause"] != "bnb_not_compiled_with_cuda"
    assert fix["fix_command"] is None, "무효한 bitsandbytes 재설치가 --yes로 실행되면 안 된다"


def test_classify_cuda_build_with_gpu_and_healthy_bnb_still_cpu() -> None:
    """같은 분기, bnb 값만 True — 판단 근거는 torch·NVML 쪽이라 결론이 같다(#72)."""
    res = _cpu_fallback(
        {
            "torch_version": "2.11.0+cu128",
            "torch_cuda_version": "12.8",
            "bnb_compiled_with_cuda": True,
            "gpu_free_mb": 9595.8,
        }
    )

    assert classify_cause(res) == "cuda_device_not_visible"


def test_cuda_device_not_visible_message_shows_current_env_value() -> None:
    """#81: fix_command는 안 붙이되, 현재 CUDA_VISIBLE_DEVICES 값을 진단 정보로 보여준다."""
    res = _cpu_fallback(
        {
            "torch_version": "2.11.0+cu128",
            "torch_cuda_version": "12.8",
            "bnb_compiled_with_cuda": False,
            "gpu_free_mb": 9595.8,
            "cuda_visible_devices": "-1",
        }
    )

    fix = suggest_fix(res)
    assert fix is not None
    assert fix["fix_command"] is None
    assert "CUDA_VISIBLE_DEVICES='-1'" in fix["message"]


def test_cuda_device_not_visible_message_notes_unset_env_value() -> None:
    """CUDA_VISIBLE_DEVICES가 아예 설정 안 됐으면 "없다"는 것과 값을 구분해 보여준다."""
    res = _cpu_fallback(
        {
            "torch_version": "2.11.0+cu128",
            "torch_cuda_version": "12.8",
            "bnb_compiled_with_cuda": False,
            "gpu_free_mb": 9595.8,
            "cuda_visible_devices": None,
        }
    )

    fix = suggest_fix(res)
    assert fix is not None
    assert fix["fix_command"] is None
    assert "설정되지 않음" in fix["message"]


def test_suggest_fix_never_imports_bitsandbytes(monkeypatch) -> None:
    """부모 프로세스는 bitsandbytes를 import하지 않는다 (#24, ADR-0002).

    원인 확인이 가장 필요한 상황("bitsandbytes import가 죽는 환경")이 곧 그 import가
    가장 위험한 상황이다. `.so` 로드 실패는 파이썬 예외가 아니라 SIGSEGV로 나므로
    `try/except`로도 막히지 않는다 — 아예 부르지 않아야 한다.
    """
    import builtins

    real_import = builtins.__import__

    def forbid_bitsandbytes(name, *args, **kwargs):
        if name.split(".")[0] == "bitsandbytes":
            raise AssertionError(f"부모 프로세스가 {name}을(를) import했다")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_bitsandbytes)

    fix = suggest_fix(
        {
            "status": "ok",
            "device": "cpu",
            "quant_backend": "bnb-4bit",
            "verdict": "FAIL",
            "reasons": ["quant_layer_device_cpu"],
            "env": {"bnb_compiled_with_cuda": False},
        }
    )

    assert fix is not None
    assert fix["cause"] == "bnb_not_compiled_with_cuda"


def test_suggest_fix_does_not_mutate_input() -> None:
    """진단 결과에 원인 조회용 값을 몰래 심지 않는다.

    예전에는 `check_result["compiled_with_cuda"]`를 직접 채워 넣었는데, 조회 함수가
    입력을 바꾸면 `--json` 출력에 canary가 재지 않은 필드가 섞여 나간다.
    """
    check_result = {
        "status": "oom",
        "verdict": "FAIL",
        "reasons": ["status_oom"],
        "env": {"bnb_compiled_with_cuda": True},
    }
    before = {**check_result, "env": dict(check_result["env"])}

    suggest_fix(check_result)

    assert check_result == before


def test_classify_falls_back_when_env_is_missing() -> None:
    """`env`를 못 받은 경우(구버전 canary·수집 실패)에도 분류가 죽지 않는다.

    "CUDA 지원 없이 빌드됨"을 **모르는 것**과 **아닌 것**은 다르다. 모를 때 재설치
    명령을 띄우면, bitsandbytes가 멀쩡한 사용자에게 엉뚱한 조치를 안내하게 된다.
    """
    base = {
        "status": "ok",
        "device": "cpu",
        "quant_backend": "bnb-4bit",
        "verdict": "FAIL",
        "reasons": ["quant_layer_device_cpu"],
    }
    envs = (
        None,
        {},
        {"bnb_compiled_with_cuda": None},
        # 자식이 아무 속성도 못 읽어 골격만 온 경우(worker._empty_env()). 여기서
        # torch_cuda_version이 None인 것은 "CPU 전용 빌드"가 아니라 "수집 실패"다 —
        # 둘을 섞으면 멀쩡한 CUDA 환경에 torch 재설치를 권하게 된다(#55).
        {
            "torch_version": None,
            "torch_cuda_version": None,
            "bnb_compiled_with_cuda": None,
            "bnb_cpu_4bit_supported": None,
        },
    )
    for env in envs:
        assert classify_cause({**base, "env": env}) == "4bit_cpu_fallback_other", env
    assert classify_cause(base) == "4bit_cpu_fallback_other"


def test_classify_and_suggest_oom() -> None:
    res = {
        "status": "oom",
        "verdict": "FAIL",
        "reasons": ["status_oom"],
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
        apply_fix({"fix_command": None, "fix_argv": None})
        apply_fix({"fix_command": "", "fix_argv": None})
        apply_fix({})
        mock_run.assert_not_called()


def test_apply_fix_executes_argv_without_reparsing_the_display_string() -> None:
    """실행은 `fix_argv`로 한다 — 표시용 문자열을 다시 파싱하지 않는다 (#52).

    Windows의 `C:\\Program Files\\...` 경로는 문자열로 조립했다가 `shlex.split`으로
    되돌리면 인용이 깨진다. 조립하지 않으면 되돌릴 일도 없다.
    """
    argv = [r"C:\Program Files\Py 3.11\python.exe", *BNB_REINSTALL_ARGS]

    with patch("subprocess.run") as mock_run:
        apply_fix({"fix_argv": argv, "fix_command": "표시용 문자열은 무시된다"})
        mock_run.assert_called_once_with(
            argv,
            capture_output=True,
            text=True,
            check=True,
        )


def test_apply_fix_falls_back_to_command_string() -> None:
    """`fix_argv`가 없는 예전 형태의 fix dict도 그대로 실행된다."""
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
    fake_raw = {"status": "import_crash", "error_log": "libbitsandbytes_cpu.so: CUDA Setup failed"}
    fake_res = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["status_import_crash"],
        "error_log": "libbitsandbytes_cpu.so: CUDA Setup failed",
    }

    with (
        patch("preflight.cli.run_canary_check", return_value=fake_raw),
        patch("preflight.cli.judge_result", return_value=fake_res),
        patch("preflight.cli.render_report"),
        patch("preflight.cli.apply_fix") as mock_apply_fix,
    ):
        result = runner.invoke(
            app,
            [
                "check",
            ],
        )
        assert result.exit_code == 1
        mock_apply_fix.assert_not_called()


def test_cli_check_with_yes_executes_fix() -> None:
    runner = CliRunner()
    fake_raw = {"status": "import_crash", "error_log": "libbitsandbytes_cpu.so: CUDA Setup failed"}
    fake_res = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["status_import_crash"],
        "error_log": "libbitsandbytes_cpu.so: CUDA Setup failed",
    }

    with (
        patch("preflight.cli.run_canary_check", return_value=fake_raw),
        patch("preflight.cli.judge_result", return_value=fake_res),
        patch("preflight.cli.render_report"),
        patch("preflight.cli.reverify", return_value=fake_res),
        patch("preflight.cli.apply_fix") as mock_apply_fix,
    ):
        result = runner.invoke(app, ["check", "--yes"])
        assert result.exit_code == 1
        mock_apply_fix.assert_called_once()
        fix_arg = mock_apply_fix.call_args[0][0]
        assert fix_arg["cause"] == "bnb_not_compiled_with_cuda"
        assert fix_arg["fix_argv"] == [sys.executable, *BNB_REINSTALL_ARGS]


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
        result = runner.invoke(app, ["check", "--yes"])
        assert result.exit_code == 0
        mock_apply_fix.assert_not_called()


def test_cli_check_with_yes_reverify_pass_exits_0() -> None:
    runner = CliRunner()
    fake_raw = {"status": "import_crash", "error_log": "libbitsandbytes_cpu.so: CUDA Setup failed"}
    fake_res_fail = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["status_import_crash"],
        "error_log": "libbitsandbytes_cpu.so: CUDA Setup failed",
    }
    fake_res_pass = {
        "status": "ok",
        "device": "cuda",
        "verdict": "PASS",
        "reasons": [],
    }

    with (
        patch("preflight.cli.run_canary_check", return_value=fake_raw),
        patch("preflight.cli.judge_result", return_value=fake_res_fail),
        patch("preflight.cli.render_report"),
        patch("preflight.cli.reverify", return_value=fake_res_pass),
        patch("preflight.cli.apply_fix") as mock_apply_fix,
    ):
        result = runner.invoke(app, ["check", "--yes"])
        assert result.exit_code == 0
        mock_apply_fix.assert_called_once()
