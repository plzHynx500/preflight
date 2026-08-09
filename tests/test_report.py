"""render_report() 출력 형식 검증. 출력 예시는 docs/contracts/cli.md 참고."""

from __future__ import annotations

import json

from preflight.canary.judge import judge_result
from preflight.report import render_report

_OK_RAW = {
    "status": "ok",
    "device": "cuda",
    "memory_delta_mb": 130.7,
    "elapsed_ms": 1.8,
    "cpu_multiplier": 19.0,
    "quant_backend": "bnb-4bit",
    "error_log": None,
}


def test_render_report_pass_text(capsys) -> None:
    result = judge_result(_OK_RAW)

    render_report([result])

    out = capsys.readouterr().out
    assert "✔" in out
    assert "FIX:" not in out
    assert "문제 없음" in out


def test_render_report_warn_cpu_multiplier_text(capsys) -> None:
    raw = {**_OK_RAW, "cpu_multiplier": 1.83}
    result = judge_result(raw)

    render_report([result])

    out = capsys.readouterr().out
    assert "⚠" in out
    assert "2배 미만" in out
    assert "1개 문제 발견" in out


def test_render_report_fail_oom_text(capsys) -> None:
    raw = {**_OK_RAW, "status": "oom", "memory_delta_mb": None, "elapsed_ms": None}
    result = judge_result(raw)

    render_report([result])

    out = capsys.readouterr().out
    assert "✖" in out
    assert "CUDA Out of Memory" in out
    # status != ok이면 timing/quant 라인은 건너뛴다.
    assert "실행 시간" not in out
    assert "bitsandbytes" not in out


def test_render_report_fail_import_crash_text(capsys) -> None:
    raw = {**_OK_RAW, "status": "import_crash", "error_log": "libcuda.so not found"}
    result = judge_result(raw)

    render_report([result])

    out = capsys.readouterr().out
    assert "import 중 크래시" in out
    assert "CUDA Out of Memory" not in out
    assert "libcuda.so not found" in out


def test_render_report_status_error_text(capsys) -> None:
    raw = {**_OK_RAW, "status": "error", "error_log": "config 조회 실패"}
    result = judge_result(raw)

    render_report([result])

    out = capsys.readouterr().out
    assert "원인 미상 오류" in out
    assert "CUDA Out of Memory" not in out
    assert "import 중 크래시" not in out


def test_render_report_quant_layer_device_cpu_fail_text(capsys) -> None:
    raw = {**_OK_RAW, "device": "cpu"}
    result = judge_result(raw)

    render_report([result])

    out = capsys.readouterr().out
    assert "device=cpu 감지 → 조용한 CPU 폴백" in out
    assert "1개 문제 발견" in out


def test_render_report_quant_fallback_is_informational_not_failure(capsys) -> None:
    """폴백 + device=cuda(#18 조합표 3행) — GPU는 정상이라 정보성 표시만 나온다."""
    raw = {**_OK_RAW, "device": "cuda", "quant_backend": "nn-linear-fallback"}
    result = judge_result(raw)
    assert result["verdict"] == "PASS"

    render_report([result])

    out = capsys.readouterr().out
    assert "nn.Linear로 대체 실행됨" in out
    assert "문제 없음" in out
    assert "✖" not in out
    assert "⚠" not in out


def test_render_report_fallback_on_cpu_is_fail_with_both_lines(capsys) -> None:
    """폴백 + device=cpu(#18 조합표 4행) — FAIL 줄과 폴백 정보 줄이 함께 나온다."""
    raw = {**_OK_RAW, "device": "cpu", "quant_backend": "nn-linear-fallback"}
    result = judge_result(raw)
    assert result["verdict"] == "FAIL"

    render_report([result])

    out = capsys.readouterr().out
    assert "✖" in out
    assert "device=cpu 감지" in out
    assert "nn.Linear로 대체 실행됨" in out
    assert "1개 문제 발견" in out


def test_render_report_with_fix_shows_fix_command(capsys) -> None:
    raw = {**_OK_RAW, "status": "oom", "memory_delta_mb": None, "elapsed_ms": None}
    result = judge_result(raw)
    result["fix"] = {
        "cause": "oom",
        "message": "CUDA Out of Memory",
        "fix_command": "pip install bitsandbytes --upgrade --force-reinstall",
    }

    render_report([result])

    out = capsys.readouterr().out
    assert "FIX: pip install bitsandbytes --upgrade --force-reinstall" in out
    assert "재확인: preflight check --yes" in out


def test_render_report_with_fix_command_none_shows_message_only(capsys) -> None:
    raw = {**_OK_RAW, "status": "oom", "memory_delta_mb": None, "elapsed_ms": None}
    result = judge_result(raw)
    result["fix"] = {
        "cause": "oom",
        "message": "CUDA Out of Memory: batch_size 축소 필요",
        "fix_command": None,
    }

    render_report([result])

    out = capsys.readouterr().out
    assert "안내: CUDA Out of Memory: batch_size 축소 필요" in out
    assert "FIX:" not in out
    assert "재확인:" not in out


def test_render_report_without_fix_key_does_not_crash(capsys) -> None:
    raw = {**_OK_RAW, "status": "oom", "memory_delta_mb": None, "elapsed_ms": None}
    result = judge_result(raw)
    assert "fix" not in result

    render_report([result])

    out = capsys.readouterr().out
    assert "FIX:" not in out


def test_render_report_json_mode_shape(capsys) -> None:
    raw = {**_OK_RAW, "status": "oom", "memory_delta_mb": None, "elapsed_ms": None}
    result = judge_result(raw)
    result["fix"] = {"cause": "oom", "message": "m", "fix_command": None}
    result["expected_memory_delta_mb"] = 100.0

    render_report([result], json_output=True)

    out = capsys.readouterr().out
    payload = json.loads(out)

    assert set(payload.keys()) == {"results", "summary", "exit_code_hint"}
    assert payload["summary"]["fail"] == 1
    assert payload["results"][0]["verdict"] == "FAIL"
    # results는 병합된 추가 키(fix, expected_memory_delta_mb)까지 그대로 통과된다.
    assert payload["results"][0]["fix"] == {"cause": "oom", "message": "m", "fix_command": None}
    assert payload["results"][0]["expected_memory_delta_mb"] == 100.0


def test_render_report_json_mode_all_pass_exit_hint_zero(capsys) -> None:
    result = judge_result(_OK_RAW)

    render_report([result], json_output=True)

    out = capsys.readouterr().out
    payload = json.loads(out)

    assert payload["exit_code_hint"] == 0
    assert payload["summary"]["fail"] == 0
    assert payload["summary"]["warn"] == 0


def test_render_report_elapsed_seconds_included_when_given_text(capsys) -> None:
    result = judge_result(_OK_RAW)

    render_report([result], elapsed_seconds=4.0)

    out = capsys.readouterr().out
    assert "소요 시간 4초" in out


def test_render_report_elapsed_seconds_omitted_when_none_text(capsys) -> None:
    result = judge_result(_OK_RAW)

    render_report([result], elapsed_seconds=None)

    out = capsys.readouterr().out
    assert "소요 시간" not in out


def test_render_report_elapsed_seconds_included_when_given_json(capsys) -> None:
    result = judge_result(_OK_RAW)

    render_report([result], json_output=True, elapsed_seconds=18.0)

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["summary"]["elapsed_seconds"] == 18.0


def test_render_report_elapsed_seconds_omitted_when_none_json(capsys) -> None:
    result = judge_result(_OK_RAW)

    render_report([result], json_output=True, elapsed_seconds=None)

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["summary"]["elapsed_seconds"] is None


def test_render_report_multiple_reasons_same_result(capsys) -> None:
    """FAIL+WARN이 함께 있는 result — 둘 다 표시되고 카운트에 반영된다."""
    raw = {**_OK_RAW, "status": "oom", "cpu_multiplier": 1.5, "memory_delta_mb": None}
    result = judge_result(raw)
    assert result["verdict"] == "FAIL"
    assert "status_oom" in result["reasons"]
    assert "cpu_multiplier_low" in result["reasons"]

    render_report([result])

    out = capsys.readouterr().out
    # status != ok이면 timing 라인 자체를 건너뛰므로, cpu_multiplier_low 메시지는
    # 표시되지 않는다 — status 라인(FAIL)만 나온다.
    assert "CUDA Out of Memory" in out
    assert "1개 문제 발견" in out


_MODEL_MODE_RAW = {
    "status": "ok",
    "device": "cuda",
    "memory_delta_mb": 8600.0,
    "elapsed_ms": None,
    "cpu_multiplier": None,  # --model 모드는 cpu_multiplier를 재지 않는다
    "quant_backend": "bnb-4bit",
    "error_log": None,
}


def test_render_report_model_mode_shows_vram_not_quant_line(capsys) -> None:
    """--model 모드(cli.md 두 번째 예시)는 quant/4bit 줄이 없고 VRAM 줄만 나온다."""
    result = judge_result(_MODEL_MODE_RAW)

    render_report([result])

    out = capsys.readouterr().out
    assert "VRAM 실측" in out
    assert "bitsandbytes" not in out


def test_render_report_model_mode_target_size_shown_when_available(capsys) -> None:
    raw = {**_MODEL_MODE_RAW, "batch_size": 2, "seq_len": 2048}
    result = judge_result(raw)

    render_report([result])

    out = capsys.readouterr().out
    assert "목표 배치 크기 적합" in out
    assert "batch=2, seq=2048" in out


def test_render_report_model_mode_target_size_omitted_when_missing(capsys) -> None:
    """batch_size/seq_len이 raw에 없으면(현재 스키마의 정상 상태) 크래시 없이 생략한다."""
    result = judge_result(_MODEL_MODE_RAW)

    render_report([result])

    out = capsys.readouterr().out
    assert "목표 배치 크기 적합" not in out
    assert "1개 모델 확인" in out
    assert "문제 없음" in out
