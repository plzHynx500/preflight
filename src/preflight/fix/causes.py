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
        env = check_result.get("env") or {}

        missing = _classify_missing_library(env)
        if missing:
            return missing

        if _BNB_IMPORT_SIGNATURE in error_log.lower():
            return "bnb_not_compiled_with_cuda"
        # 라이브러리는 설치돼 있는데 로드가 깨진 경우다 — CUDA 런타임·드라이버 계열
        # (c10_cuda.dll, libcudart 등)이 여기로 떨어진다. 어느 import에서 죽었는지는
        # 아직 모르므로(#93의 후속안 A) fix_command가 없는 일반 원인으로 둔다.
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
        # 모델 체크가 쓰는 transformers는 `_import_canary_stack()`이 아니라 `_run()`
        # 안에서 import된다 — 없으면 `import_crash`가 아니라 여기로 떨어진다(실측).
        # 설치 여부는 같은 방식으로 판정한다.
        missing = _classify_missing_library(check_result.get("env") or {})
        if missing:
            return missing
        return "unknown_error"

    # 6. WARN 항목
    # 미설치는 성능 경고보다 먼저다 — pip 한 줄로 끝나는, 가장 조치가 명확한 문제다.
    if "qlora_stack_not_installed" in reasons:
        return "qlora_stack_not_installed"
    if "memory_delta_high" in reasons:
        return "memory_delta_high"
    if "cpu_multiplier_low" in reasons:
        return "cpu_multiplier_low"

    return "unknown"


def _classify_missing_library(env: dict) -> str | None:
    """설치되지 않은 라이브러리가 있으면 그 원인 코드, 아니면 None.

    로그 문자열보다 **설치 여부를 먼저** 본다. `env.*_installed`는 자식이
    `find_spec`으로 읽은 환경 사실이라(#44) 문구가 바뀌어도 흔들리지 않고,
    트레이스백이 아예 없는 경우(네이티브 즉사)에도 유효하다.

    **`False`일 때만 단정한다** — `None`은 "설치 안 됨"이 아니라 "못 읽었다"이므로
    호출 측의 기존 분류로 흘려보낸다("모르는 것"과 "아닌 것"을 섞지 않는다).

    `status`가 `import_crash`인 경우와 `error`인 경우 모두에서 쓴다. torch는
    `_import_canary_stack()`에서, transformers는 `_run()` 안에서 import되므로
    **없을 때 떨어지는 status가 서로 다르다** — 판정은 같아야 한다.

    bitsandbytes는 여기서 다루지 않는다. 없어도 `_try_import_bitsandbytes()`가
    삼키고 `nn.Linear`로 폴백하므로 실패 경로로 오지 않는다 — 그 사실은 화면의
    폴백 안내에서 `env.bitsandbytes_installed`로 알린다(#60).
    """
    if env.get("torch_installed") is False:
        # GPU가 실제로 보일 때만 CUDA 빌드 설치를 권한다 — GPU가 없는 기계(AMD 등)에서
        # 2GB가 넘는 CUDA 휠을 받아봐야 아무것도 달라지지 않는다.
        # `torch_cpu_only_build` / `..._no_gpu`와 같은 갈림이다(#55).
        gpu_seen = env.get("gpu_free_mb") is not None
        return "torch_not_installed" if gpu_seen else "torch_not_installed_no_gpu"
    if env.get("transformers_installed") is False:
        return "transformers_not_installed"
    return None


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
