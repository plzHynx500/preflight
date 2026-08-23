"""판정 로직(PASS/WARN/FAIL). docs/contracts/canary-api.md의 judge_result 계약을 구현한다.

두 임계값은 매직넘버로 우선 채택한 것이며, 실측 데이터가 쌓이면 조정될 수 있다.
"""

from __future__ import annotations

VRAM_HEADROOM_WARN_RATIO = 0.9
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

    # memory_delta_mb를 "예측 대비 15% 이탈"로 보던 원래 설계는 probe 기반
    # 외삽(§7, 향후 확장)이 있어야 "예측값"이 생기는데 MVP에는 그게 없어 사실상
    # 죽어있는 조건이었다. 2026-08-06 회의(안건 1)에서 "예측값과 비교" 대신
    # "지금 가용한 VRAM 대비 canary가 이미 얼마나 먹었는가"로 기준을 바꾸기로
    # 했다 — 예측 모델 없이도 이 GPU에서 당장 room이 부족한지 알 수 있고,
    # 사용자에게도 "왜 위험한지" 훨씬 직관적으로 전달된다. reason 이름
    # "memory_delta_high"는 그대로 유지한다("판정 항목 5개 중 하나의 의미가
    # 바뀐 것"으로 취급 — 새 reason을 늘리지 않는다).
    #
    # gpu_free_mb는 canary를 "돌리지 않고도" 알 수 있는 값(gpu.query_gpu_state()가
    # canary 기동 직전 부모 프로세스에서 조회)이라 env 서브딕트에 둔다 — 이 값이
    # 없으면(NVML 조회 실패 등) 판정을 건너뛸 뿐 FAIL로 취급하지 않는다.
    memory_delta_mb = raw.get("memory_delta_mb")
    gpu_free_mb = (raw.get("env") or {}).get("gpu_free_mb")
    if memory_delta_mb is not None and gpu_free_mb is not None and gpu_free_mb > 0:
        headroom_ratio = memory_delta_mb / gpu_free_mb
        if headroom_ratio >= VRAM_HEADROOM_WARN_RATIO:
            reasons.append("memory_delta_high")

    cpu_multiplier = raw.get("cpu_multiplier")
    if cpu_multiplier is not None and cpu_multiplier < CPU_MULTIPLIER_WARN_THRESHOLD:
        reasons.append("cpu_multiplier_low")

    # "quant_fallback"은 판정에 영향을 주지 않는 정보성 표시다(리포트 노출 목적).
    # reasons 전체가 아니라 FAIL/WARN 사유 집합에 속하는지로만 verdict를 정한다.
    # QLoRA 학습에 필요한 라이브러리가 없으면 화면과 판정에 남긴다(#117).
    #
    # canary는 bitsandbytes가 없으면 nn.Linear로 폴백하고 peft는 아예 쓰지 않아
    # **둘 다 없어도 여기까지 무사히 온다.** 하지만 사용자가 실제로 QLoRA 학습을
    # 시작하면 `from_pretrained(quantization_config=...)`와 `from peft import ...`가
    # 첫 줄에서 ImportError로 죽는다(실측) — 우리가 PASS를 내주면 거짓 안심이다.
    #
    # **FAIL이 아니라 WARN인 이유**: torch 미설치와 달리 "더 무거운 길"이 남아
    # 있다. bnb만 없으면 양자화 없는 LoRA가, peft만 없으면 full finetuning이
    # 가능하다. 다만 우리가 잰 숫자가 그 사용자가 실제로 할 수 있는 일을
    # 대표하지 않으므로 PASS도 아니다.
    #
    # `False`(설치 안 됨 확정)일 때만 잡는다 — `None`은 "못 읽었다"이므로
    # 단정하지 않는다.
    env = raw.get("env") or {}
    if env.get("bitsandbytes_installed") is False or env.get("peft_installed") is False:
        reasons.append("qlora_stack_not_installed")

    fail_reasons = {"status_oom", "status_import_crash", "status_error", "quant_layer_device_cpu"}
    warn_reasons = {"memory_delta_high", "cpu_multiplier_low", "qlora_stack_not_installed"}
    if any(reason in fail_reasons for reason in reasons):
        verdict = "FAIL"
    elif any(reason in warn_reasons for reason in reasons):
        verdict = "WARN"
    else:
        verdict = "PASS"

    return {**raw, "verdict": verdict, "reasons": reasons}
