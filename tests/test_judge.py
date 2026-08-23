"""판정 기준은 docs/contracts/canary-api.md 참고."""

import pytest

from preflight.canary.judge import judge_result

_OK_RAW = {
    "status": "ok",
    "device": "cuda",
    "memory_delta_mb": 130.7,
    "elapsed_ms": 1.8,
    "cpu_multiplier": 19.0,
    "quant_backend": "bnb-4bit",
    "error_log": None,
}


def test_judge_result_pass() -> None:
    result = judge_result(_OK_RAW)

    assert result["verdict"] == "PASS"
    assert result["reasons"] == []
    # 원본 필드는 그대로 보존된다.
    assert result["memory_delta_mb"] == 130.7


def test_oom_is_fail() -> None:
    raw = {**_OK_RAW, "status": "oom", "memory_delta_mb": None, "elapsed_ms": None}

    result = judge_result(raw)

    assert result["verdict"] == "FAIL"
    assert "status_oom" in result["reasons"]


def test_import_crash_is_fail() -> None:
    raw = {**_OK_RAW, "status": "import_crash"}

    result = judge_result(raw)

    assert result["verdict"] == "FAIL"
    assert "status_import_crash" in result["reasons"]


def test_error_status_is_fail() -> None:
    """import_crash/oom 어느 쪽으로도 확정 못 하는 실패(예: 모델명 오타)도 FAIL이다."""
    raw = {**_OK_RAW, "status": "error", "error_log": "config 조회 실패"}

    result = judge_result(raw)

    assert result["verdict"] == "FAIL"
    assert "status_error" in result["reasons"]


def test_4bit_layer_on_cpu_is_fail() -> None:
    raw = {**_OK_RAW, "device": "cpu"}

    result = judge_result(raw)

    assert result["verdict"] == "FAIL"
    assert "quant_layer_device_cpu" in result["reasons"]


def test_bnb_4bit_on_cuda_is_pass() -> None:
    """quant_backend x device 조합표 1행 (#18)."""
    raw = {**_OK_RAW, "device": "cuda", "quant_backend": "bnb-4bit"}

    result = judge_result(raw)

    assert result["reasons"] == []
    assert result["verdict"] == "PASS"


def test_bnb_4bit_on_cpu_is_fail() -> None:
    """조합표 2행 — 기존에도 잡히던 케이스, 계속 잡혀야 한다."""
    raw = {**_OK_RAW, "device": "cpu", "quant_backend": "bnb-4bit"}

    result = judge_result(raw)

    assert "quant_layer_device_cpu" in result["reasons"]
    assert "quant_fallback" not in result["reasons"]
    assert result["verdict"] == "FAIL"


def test_fallback_on_cuda_is_pass_with_info_reason() -> None:
    """조합표 3행 — 4bit은 못 쓰지만 GPU 자체는 정상이면 PASS + 정보성 표시만.

    quant_fallback은 verdict에 영향을 주지 않는 정보성 reason이다.
    """
    raw = {**_OK_RAW, "device": "cuda", "quant_backend": "nn-linear-fallback"}

    result = judge_result(raw)

    assert "quant_layer_device_cpu" not in result["reasons"]
    assert "quant_fallback" in result["reasons"]
    assert result["verdict"] == "PASS"


def test_fallback_on_cpu_is_fail() -> None:
    """조합표 4행(#18에서 발견된 구멍) — GPU도 4bit도 안 되면 반드시 FAIL이어야 한다.

    수정 전에는 quant_layer_device_cpu가 quant_backend=="bnb-4bit"일 때만
    평가돼서, 이 조합(4bit 레이어 자체가 없는 환경)은 아무 FAIL도 안 걸리고
    PASS로 새 나갔다 — device=="cpu"는 quant_backend와 무관하게 그 자체로
    FAIL이어야 한다.
    """
    raw = {**_OK_RAW, "device": "cpu", "quant_backend": "nn-linear-fallback"}

    result = judge_result(raw)

    assert "quant_layer_device_cpu" in result["reasons"]
    assert "quant_fallback" in result["reasons"]
    assert result["verdict"] == "FAIL"


def test_cpu_multiplier_below_threshold_is_warn() -> None:
    raw = {**_OK_RAW, "cpu_multiplier": 1.83}

    result = judge_result(raw)

    assert result["verdict"] == "WARN"
    assert "cpu_multiplier_low" in result["reasons"]


def test_cpu_multiplier_none_is_not_evaluated() -> None:
    """--model 실행은 cpu_multiplier를 재지 않는다(None) — 판정 대상에서 빠진다."""
    raw = {**_OK_RAW, "cpu_multiplier": None}

    result = judge_result(raw)

    assert "cpu_multiplier_low" not in result["reasons"]
    assert result["verdict"] == "PASS"


def test_memory_delta_high_when_headroom_mostly_consumed() -> None:
    """2026-08-06 회의(안건 1): "예측 대비 15% 이탈" 대신 "가용 VRAM 대비 소모율"로 판정한다.

    gpu_free_mb(canary 기동 직전 조회된 가용 VRAM)의 90% 이상을 canary
    실행만으로 소모했으면 실제 학습에서 OOM 위험이 크다고 보고 WARN이다.
    """
    raw = {**_OK_RAW, "memory_delta_mb": 950.0, "env": {"gpu_free_mb": 1000.0}}

    result = judge_result(raw)

    assert result["verdict"] == "WARN"
    assert "memory_delta_high" in result["reasons"]


def test_memory_delta_within_headroom_is_pass() -> None:
    raw = {**_OK_RAW, "memory_delta_mb": 300.0, "env": {"gpu_free_mb": 1000.0}}

    result = judge_result(raw)

    assert "memory_delta_high" not in result["reasons"]
    assert result["verdict"] == "PASS"


def test_memory_delta_high_skipped_when_gpu_state_unavailable() -> None:
    """gpu_free_mb를 못 구했으면(NVML 조회 실패 등) 이 WARN은 조용히 건너뛴다 — FAIL이 아니다."""
    raw = {**_OK_RAW, "memory_delta_mb": 99999.0, "env": {}}

    result = judge_result(raw)

    assert "memory_delta_high" not in result["reasons"]
    assert result["verdict"] == "PASS"


def test_memory_delta_high_skipped_when_env_missing() -> None:
    """env 키 자체가 없는(구버전 raw) 입력도 예외 없이 건너뛴다."""
    raw = {**_OK_RAW, "memory_delta_mb": 99999.0}
    raw.pop("env", None)

    result = judge_result(raw)

    assert "memory_delta_high" not in result["reasons"]
    assert result["verdict"] == "PASS"


def test_multiple_reasons_combine() -> None:
    """FAIL 사유가 있으면 WARN 사유가 같이 있어도 최종 verdict는 FAIL이다."""
    raw = {**_OK_RAW, "status": "oom", "cpu_multiplier": 1.5, "memory_delta_mb": None}

    result = judge_result(raw)

    assert result["verdict"] == "FAIL"
    assert "status_oom" in result["reasons"]
    assert "cpu_multiplier_low" in result["reasons"]


def test_cpu_multiplier_exactly_at_threshold_is_pass() -> None:
    """경계값(2배 정각)은 미만이 아니므로 WARN이 아니다."""
    raw = {**_OK_RAW, "cpu_multiplier": 2.0}

    result = judge_result(raw)

    assert "cpu_multiplier_low" not in result["reasons"]
    assert result["verdict"] == "PASS"


def test_memory_delta_exactly_at_threshold_is_warn() -> None:
    """경계값(가용 VRAM의 90% 정각)은 이상이므로 WARN이다."""
    raw = {**_OK_RAW, "memory_delta_mb": 900.0, "env": {"gpu_free_mb": 1000.0}}

    result = judge_result(raw)

    assert "memory_delta_high" in result["reasons"]
    assert result["verdict"] == "WARN"


def test_device_cpu_fail_combined_with_unrelated_warn() -> None:
    """4bit device=cpu(FAIL)와 cpu_multiplier 낮음(WARN)이 동시에 있어도 FAIL이 이긴다."""
    raw = {**_OK_RAW, "device": "cpu", "cpu_multiplier": 1.5}

    result = judge_result(raw)

    assert result["verdict"] == "FAIL"
    assert "quant_layer_device_cpu" in result["reasons"]
    assert "cpu_multiplier_low" in result["reasons"]


# --- QLoRA 스택 미설치는 WARN 이다 (#117) ---


def _fallback_raw(env: dict) -> dict:
    return {
        "status": "ok",
        "device": "cuda",
        "quant_backend": "nn-linear-fallback",
        "env": env,
    }


@pytest.mark.parametrize(
    "env",
    [
        {"bitsandbytes_installed": False, "peft_installed": True},
        {"bitsandbytes_installed": True, "peft_installed": False},
        {"bitsandbytes_installed": False, "peft_installed": False},
    ],
)
def test_missing_qlora_stack_is_warn_not_pass(env: dict) -> None:
    """설치가 안 됐으면 PASS 를 내주지 않는다 (#117).

    canary 는 bitsandbytes 없이 nn.Linear 로 폴백하고 peft 는 아예 안 써서 여기까지
    무사히 온다. 하지만 사용자가 실제로 QLoRA 학습을 시작하면 첫 줄에서
    ImportError 로 죽는다 — PASS 는 거짓 안심이다.
    """
    result = judge_result(_fallback_raw(env))

    assert result["verdict"] == "WARN"
    assert "qlora_stack_not_installed" in result["reasons"]


def test_installed_qlora_stack_stays_pass() -> None:
    env = {"bitsandbytes_installed": True, "peft_installed": True}

    assert judge_result(_fallback_raw(env))["verdict"] == "PASS"


@pytest.mark.parametrize("env", [{}, {"bitsandbytes_installed": None, "peft_installed": None}])
def test_unknown_install_state_does_not_warn(env: dict) -> None:
    """`None`(못 읽음)을 "설치 안 됨"으로 단정하지 않는다 — 멀쩡한 환경에 경고를 만들지 않는다."""
    result = judge_result(_fallback_raw(env))

    assert result["verdict"] == "PASS"
    assert "qlora_stack_not_installed" not in result["reasons"]


def test_real_failure_still_wins_over_the_warn() -> None:
    """FAIL 이 있으면 그쪽이 이긴다 — 미설치 경고가 실패를 가리지 않는다."""
    raw = _fallback_raw({"bitsandbytes_installed": False})
    raw["device"] = "cpu"

    assert judge_result(raw)["verdict"] == "FAIL"
