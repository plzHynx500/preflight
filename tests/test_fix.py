from __future__ import annotations

from preflight.fix.causes import classify_cause
from preflight.fix.executor import suggest_fix


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
        "reasons": ["import_crash"],
        "error_log": "libbitsandbytes_cpu.so CUDA Setup failed",
    }
    assert classify_cause(res1) == "bnb_not_compiled_with_cuda"
    fix1 = suggest_fix(res1)
    assert fix1 is not None
    assert fix1["cause"] == "bnb_not_compiled_with_cuda"
    assert fix1["fix_command"] == "pip install bitsandbytes --upgrade --force-reinstall"

    # case 2: 4bit cpu fallback with compiled_with_cuda=False
    res2 = {
        "status": "ok",
        "verdict": "FAIL",
        "reasons": ["4bit_cpu_fallback"],
        "compiled_with_cuda": False,
    }
    assert classify_cause(res2) == "bnb_not_compiled_with_cuda"
    fix2 = suggest_fix(res2)
    assert fix2 is not None
    assert fix2["cause"] == "bnb_not_compiled_with_cuda"
    assert fix2["fix_command"] == "pip install bitsandbytes --upgrade --force-reinstall"


def test_classify_and_suggest_import_general() -> None:
    res = {
        "status": "import_crash",
        "verdict": "FAIL",
        "reasons": ["import_crash"],
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
        "verdict": "FAIL",
        "reasons": ["4bit_cpu_fallback"],
        "compiled_with_cuda": True,
    }
    assert classify_cause(res) == "4bit_cpu_fallback_other"
    fix = suggest_fix(res)
    assert fix is not None
    assert fix["cause"] == "4bit_cpu_fallback_other"
    assert fix["fix_command"] is None


def test_classify_and_suggest_oom() -> None:
    res = {
        "status": "oom",
        "verdict": "FAIL",
        "reasons": ["oom"],
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
