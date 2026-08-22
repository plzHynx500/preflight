"""원인 시그니처 → 원인 분류. 안정된 파이썬 속성(torch.version.cuda,
sys.version_info 등)으로 판별하고 nvidia-smi 텍스트 파싱은 쓰지 않는다
(docs/architecture.md §5 MODULE-01 참고).

FIX 문구 자체는 초안 단계다 — docs/architecture.md §6-04 표 참고.
"""

from __future__ import annotations

# import_crash 로그에서 "bitsandbytes 문제"로 볼 수 있는 유일한 시그니처.
# 예전에는 `"cuda" in error_lower`도 함께 봤는데, 로그에 cuda라는 글자가 있기만
# 하면(c10_cuda.dll 없음, libcudart.so.12 없음 등 torch·CUDA 런타임 쪽 문제까지)
# 전부 bitsandbytes 탓으로 돌려 `--yes`가 무관한 재설치를 실제로 실행했다(#51).
# `libbitsandbytes_cpu.so` 같은 파일명도 이 부분문자열에 걸린다.
_BNB_IMPORT_SIGNATURE = "bitsandbytes"


def classify_cause(check_result: dict) -> str:
    """Canary check_result 딕셔너리에서 가장 중요한 대표 원인(cause code)을 분류한다.

    우선순위: FAIL (import_crash > oom > 4bit_cpu_fallback) > WARN (memory_delta_high > cpu_multiplier_low)
    """
    verdict = check_result.get("verdict")
    status = check_result.get("status", "ok")
    reasons = check_result.get("reasons", [])
    error_log = str(check_result.get("error_log") or "")

    # 1. PASS 처리
    if verdict == "PASS":
        return "pass"

    # 2. FAIL - import_crash
    if status == "import_crash" or "status_import_crash" in reasons:
        if _BNB_IMPORT_SIGNATURE in error_log.lower():
            return "bnb_not_compiled_with_cuda"
        # CUDA 런타임·드라이버 계열(c10_cuda.dll, libcudart 등)도 여기로 떨어진다.
        # 전용 원인을 새로 만드는 건 #44에서 다룬다 — 지금은 "틀린 명령을 실행하지
        # 않는 것"이 목적이라 fix_command가 없는 일반 원인으로 두는 편이 안전하다.
        return "import_crash_general"

    # 3. FAIL - oom
    if status == "oom" or "status_oom" in reasons:
        return "oom"

    # 4. FAIL - 4bit cpu fallback
    is_cpu_fallback = "quant_layer_device_cpu" in reasons or (
        check_result.get("quant_backend") == "bnb-4bit" and check_result.get("device") == "cpu"
    )
    if is_cpu_fallback:
        return _classify_cpu_fallback(check_result.get("env") or {})

    # 5. status == "error"
    if status == "error":
        return "unknown_error"

    # 6. WARN 항목
    if "memory_delta_high" in reasons:
        return "memory_delta_high"
    if "cpu_multiplier_low" in reasons:
        return "cpu_multiplier_low"

    return "unknown"


def _classify_cpu_fallback(env: dict) -> str:
    """4bit 레이어가 cpu로 떨어진 이유를 `env`의 환경 사실로 가른다.

    canary 자식이 채워 보낸 값만 읽는다 (Issue #19). 부모가 직접 `import
    bitsandbytes`로 확인하면, 진단 대상인 "bnb import가 죽는 환경"에서 CLI까지
    함께 죽어 FR-03 격리가 무너진다.

    **판정 순서가 곧 신호의 신뢰도 순서다.** 아래로 갈수록 근거가 약하다:

    1. `torch_cuda_version` — torch 자신이 CUDA 빌드인지. 가장 확실하고, 이게
       None이면 GPU가 있든 없든 CUDA 연산 자체가 불가능하다(#55).
    2. `gpu_free_mb` — NVML이 NVIDIA GPU를 실제로 조회했는지. 부모가 얹는 값이라
       torch와 독립적이다(없으면 GPU/드라이버가 없거나 NVML 조회 실패).
    3. `bnb_compiled_with_cuda` — **더 이상 단독 판단 근거로 쓰지 않는다.**
       bitsandbytes 0.50에서 이 값은 빌드 속성이 아니라 "런타임에 CUDA 장치가
       보이는가"를 반영해서, GPU가 가려졌을 뿐인 환경(CUDA_VISIBLE_DEVICES=-1)에도
       False가 나온다 — 그걸 빌드 문제로 읽고 재설치를 권하면 아무것도 안 고쳐진다
       (#72, RTX 4070 Ti·bnb 0.50.0 실측). 위 두 신호를 아예 못 읽은 경우에만
       마지막 폴백으로 참고한다.

    "모르는 것"과 "아닌 것"을 섞지 않는다 — `torch_version`조차 못 읽었으면
    `torch_cuda_version`이 None인 것은 "CPU 전용 빌드"가 아니라 "수집 실패"다.
    """
    torch_known = env.get("torch_version") is not None
    is_cuda_build = env.get("torch_cuda_version") is not None
    gpu_seen = env.get("gpu_free_mb") is not None

    if torch_known and not is_cuda_build:
        # 설치된 torch가 CPU 전용 빌드다("2.13.0+cpu"). 가장 흔한 초보 실수이고,
        # 여기서 "CUDA 메모리 부족"이라고 안내하면 사용자를 정반대 방향(배치 축소)으로
        # 보낸다(#55). NVIDIA GPU가 실제로 보일 때만 재설치 명령을 붙인다 — GPU가
        # 없는 기계(AMD 등)에서 CUDA 빌드 torch를 받아봐야 아무것도 달라지지 않는다.
        return "torch_cpu_only_build" if gpu_seen else "torch_cpu_only_build_no_gpu"

    if torch_known and is_cuda_build:
        if not gpu_seen:
            return "no_nvidia_gpu_or_driver"
        # torch는 CUDA 빌드고 NVML도 GPU를 정상 조회했는데 연산은 cpu로 갔다 —
        # 장치가 프로세스에 안 보이는 상태다(CUDA_VISIBLE_DEVICES, 드라이버/런타임
        # 불일치 등). bitsandbytes 재설치로는 해결되지 않는다(#72).
        return "cuda_device_not_visible"

    if env.get("bnb_compiled_with_cuda") is False:
        return "bnb_not_compiled_with_cuda"

    return "4bit_cpu_fallback_other"
