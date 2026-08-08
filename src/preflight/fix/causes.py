"""원인 시그니처 → 원인 분류. 안정된 파이썬 속성(torch.version.cuda,
bitsandbytes.cextension.lib.compiled_with_cuda, sys.version_info 등)으로 판별하고
nvidia-smi 텍스트 파싱은 쓰지 않는다 (docs/architecture.md §5 MODULE-01 참고).

FIX 문구 자체는 초안 단계다 — docs/architecture.md §6-04 표 참고.
"""

from __future__ import annotations


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
        error_lower = error_log.lower()
        if "bitsandbytes" in error_lower or "cuda" in error_lower:
            return "bnb_not_compiled_with_cuda"
        return "import_crash_general"

    # 3. FAIL - oom
    if status == "oom" or "status_oom" in reasons:
        return "oom"

    # 4. FAIL - 4bit cpu fallback
    is_cpu_fallback = "quant_layer_device_cpu" in reasons or (
        check_result.get("quant_backend") == "bnb-4bit" and check_result.get("device") == "cpu"
    )
    if is_cpu_fallback:
        # canary 자식이 채워 보낸 환경 속성을 읽는다 (Issue #19). 부모가 직접
        # `import bitsandbytes`로 확인하면, 진단 대상인 "bnb import가 죽는 환경"에서
        # CLI까지 함께 죽어 FR-03 격리가 무너진다.
        env = check_result.get("env") or {}
        if env.get("bnb_compiled_with_cuda") is False:
            return "bnb_not_compiled_with_cuda"
        return "4bit_cpu_fallback_other"

    # 5. status == "error"
    if status == "error":
        return "unknown_error"

    # 6. WARN 항목
    if "memory_delta_high" in reasons:
        return "memory_delta_high"
    if "cpu_multiplier_low" in reasons:
        return "cpu_multiplier_low"

    return "unknown"
