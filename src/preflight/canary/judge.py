"""판정 로직(PASS/WARN/FAIL). docs/contracts/canary-api.md의 judge_result 계약을 구현한다.

두 임계값은 매직넘버로 우선 채택한 것이며, 실측 데이터가 쌓이면 조정될 수 있다.
"""

from __future__ import annotations

MEMORY_DELTA_WARN_PCT = 15
CPU_MULTIPLIER_WARN_THRESHOLD = 2

FAIL_STATUSES = ("oom", "import_crash", "error")


def judge_result(raw: dict) -> dict:
    """원시 측정값(raw)에 verdict·reasons를 얹는다. 원인 분류는 하지 않는다.

    판정 기준은 docs/contracts/canary-api.md "judge_result" 참고.
    """
    reasons: list[str] = []

    # status가 이미 실패를 뜻하는 세 가지 중 하나면 그 자체로 FAIL이다.
    # "error"는 오타로 인한 config 조회 실패 등, oom/import_crash 어느 쪽으로도
    # 확정할 수 없는 실패를 담는 상영님 쪽 신규 필드 — 원인분류(suggest_fix)가
    # 엉뚱한 명령을 내지 않도록 여기서도 FAIL로만 취급하고 원인은 따지지 않는다.
    if raw.get("status") in FAIL_STATUSES:
        reasons.append(f"status_{raw['status']}")

    # device=cpu는 quant_backend와 무관하게 그 자체로 FAIL이다 — 기본 체크의
    # 목적 자체가 "GPU/드라이버/CUDA 체인이 물리적으로 살아있는가"라서(architecture.md
    # §3), 4bit 레이어가 있든 없든 GPU를 못 타면 실패다. 원래는 quant_backend가
    # bnb-4bit일 때만 이 조건을 봤는데, 그러면 bitsandbytes조차 없어 4bit 레이어
    # 자체가 없는 환경(quant_backend="nn-linear-fallback")은 이 규칙이 발동을
    # 못 해 device=cpu인 채로 PASS가 나가는 구멍이 있었다(#18, 상영님 지적).
    # reason 이름은 report.py(#16)와의 호환을 위해 기존 이름을 유지한다.
    if raw.get("device") == "cpu":
        reasons.append("quant_layer_device_cpu")
    # quant_fallback은 device와 별개로, 폴백이 있었다는 사실 자체를 항상 알린다
    # (판정에는 영향 없는 정보성 — report.py가 화면에 표시).
    if raw.get("quant_backend") == "nn-linear-fallback":
        reasons.append("quant_fallback")

    # memory_delta_mb를 예측치와 비교하는 WARN 조건은 probe 기반 외삽(§7, 향후
    # 확장)이 있어야 "예측값"이 생긴다. MVP에는 그 예측치가 없어 지금은 이 값이
    # 전달될 때만(향후 raw에 added) 평가한다 — 미결 사항은 Notion "!내부용!
    # 논의사항" §WARN 트리거 조건 참고.
    expected_mb = raw.get("expected_memory_delta_mb")
    memory_delta_mb = raw.get("memory_delta_mb")
    if expected_mb is not None and memory_delta_mb is not None and expected_mb > 0:
        deviation_pct = abs(memory_delta_mb - expected_mb) / expected_mb * 100
        if deviation_pct >= MEMORY_DELTA_WARN_PCT:
            reasons.append("memory_delta_high")

    cpu_multiplier = raw.get("cpu_multiplier")
    if cpu_multiplier is not None and cpu_multiplier < CPU_MULTIPLIER_WARN_THRESHOLD:
        reasons.append("cpu_multiplier_low")

    # "quant_fallback"은 판정에 영향을 주지 않는 정보성 표시다(리포트 노출 목적).
    # reasons 전체가 아니라 FAIL/WARN 사유 집합에 속하는지로만 verdict를 정한다.
    fail_reasons = {"status_oom", "status_import_crash", "status_error", "quant_layer_device_cpu"}
    warn_reasons = {"memory_delta_high", "cpu_multiplier_low"}
    if any(reason in fail_reasons for reason in reasons):
        verdict = "FAIL"
    elif any(reason in warn_reasons for reason in reasons):
        verdict = "WARN"
    else:
        verdict = "PASS"

    return {**raw, "verdict": verdict, "reasons": reasons}
