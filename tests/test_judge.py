"""판정 기준은 docs/contracts/canary-api.md 참고."""

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


def test_memory_delta_deviation_is_warn_when_expected_value_present() -> None:
    """예측치(expected_memory_delta_mb)가 있을 때만 15% 이탈 여부를 평가한다.

    MVP에는 probe 기반 외삽이 없어 이 필드가 보통 없다(§7 향후 확장) — 있을 때만
    평가하도록 만들어, 필드가 추가된 뒤에도 그대로 쓸 수 있게 해둔다.
    """
    raw = {**_OK_RAW, "memory_delta_mb": 150.0, "expected_memory_delta_mb": 100.0}

    result = judge_result(raw)

    assert result["verdict"] == "WARN"
    assert "memory_delta_high" in result["reasons"]


def test_memory_delta_within_tolerance_is_pass() -> None:
    raw = {**_OK_RAW, "memory_delta_mb": 105.0, "expected_memory_delta_mb": 100.0}

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
    """경계값(15% 정각)은 이상이므로 WARN이다."""
    raw = {**_OK_RAW, "memory_delta_mb": 115.0, "expected_memory_delta_mb": 100.0}

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
