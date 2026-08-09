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
    # model_name이 --model 모드임을 확정하는 유일한 신호다(cpu_multiplier is
    # None만으로는 GPU 없는 기본 체크와 구분이 안 된다 — 아래
    # test_render_report_basic_check_without_gpu_is_not_model_mode 참고).
    "model_name": "meta-llama/Llama-3.1-8B",
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


def test_render_report_model_mode_vram_line_shows_free_and_total_from_env(capsys) -> None:
    """VRAM 줄은 env.gpu_free_mb/gpu_total_mb(둘 다 MB)를 판정과 같은 숫자로 보여준다.

    이전에는 최상위 total_vram_gb(아무도 안 채움)를 찾아 "/ 12GB 가용"이 항상
    사라졌고, 채워졌더라도 judge_result가 WARN 판정에 쓰는 free 기준과 다른
    total 기준이라 화면·판정 숫자가 어긋났다(PR #26 리뷰, 상영님 지적).
    """
    raw = {**_MODEL_MODE_RAW, "env": {"gpu_free_mb": 9420.0, "gpu_total_mb": 12282.0}}
    result = judge_result(raw)

    render_report([result])

    out = capsys.readouterr().out
    assert "8.4GB / 9.2GB 가용 (총 12GB)" in out


def test_render_report_model_mode_vram_falls_back_when_env_missing(capsys) -> None:
    """env가 없으면(cli.py 미배선·NVML 실패 등) 크래시 없이 "GB 사용"만 보여준다."""
    result = judge_result(_MODEL_MODE_RAW)

    render_report([result])

    out = capsys.readouterr().out
    assert "8.4GB 사용" in out
    assert "가용" not in out


def test_render_report_memory_delta_high_shows_warning_line(capsys) -> None:
    """judge.py의 memory_delta_high WARN이 화면·문제 카운트에 실제로 반영된다.

    이전에는 --model 모드의 VRAM 줄이 이 reason과 무관하게 항상 초록 ✔라
    WARN이 떠도 화면에는 안 보이는 문제가 있었다(PR #26 작업 중 자체 발견).
    """
    raw = {
        **_MODEL_MODE_RAW,
        "memory_delta_mb": 9000.0,
        "env": {"gpu_free_mb": 9500.0, "gpu_total_mb": 12282.0},
    }
    result = judge_result(raw)
    assert result["verdict"] == "WARN"
    assert "memory_delta_high" in result["reasons"]

    render_report([result])

    out = capsys.readouterr().out
    assert "⚠" in out
    assert "VRAM 여유" in out
    assert "OOM 위험" in out
    assert "1개 문제 발견" in out


def test_render_report_single_result_has_no_group_label(capsys) -> None:
    """결과가 1개뿐이면(기존 출력) 표제를 붙이지 않는다 — 하위 호환."""
    result = judge_result(_OK_RAW)

    render_report([result])

    out = capsys.readouterr().out
    assert "기본 체크" not in out
    assert "모델 체크" not in out


def test_render_report_two_results_get_group_labels(capsys) -> None:
    """--model이 주어지면 기본 체크 + 모델 체크 결과 2개가 순서대로 라벨과 함께 나온다."""
    basic_result = judge_result(_OK_RAW)
    model_result = judge_result({**_MODEL_MODE_RAW, "model_name": "meta-llama/Llama-3.1-8B"})

    render_report([basic_result, model_result])

    out = capsys.readouterr().out
    basic_idx = out.index("기본 체크")
    model_idx = out.index("모델 체크: meta-llama/Llama-3.1-8B")
    assert basic_idx < model_idx
    assert "VRAM 실측" in out
    assert "bitsandbytes 4bit 레이어" in out


def test_render_report_basic_check_without_gpu_is_not_model_mode(capsys) -> None:
    """GPU 없는 기본 체크(cpu_multiplier=None)를 --model 체크로 오인하지 않는다.

    worker.py는 device != "cuda"면 CPU 배속 비교 자체를 생략해 cpu_multiplier가
    None으로 남는다 — 예전 `_is_model_mode()`(cpu_multiplier is None으로만
    추론)는 이걸 --model 체크로 오인해 "4bit 레이어 device=cpu" FAIL 줄이
    안 그려지는 분기로 새 버렸다. 판정은 FAIL인데 화면은 "문제 없음"으로
    보이는 #18 재발 버그였다(PR #26 리뷰, 상영님 지적). model_name이 없으면
    cpu_multiplier와 무관하게 기본 체크로 취급해야 한다.
    """
    raw = {**_OK_RAW, "device": "cpu", "cpu_multiplier": None}
    result = judge_result(raw)
    assert result["verdict"] == "FAIL"

    render_report([result])

    out = capsys.readouterr().out
    assert "device=cpu 감지 → 조용한 CPU 폴백" in out
    assert "1개 문제 발견" in out
    assert "VRAM 실측" not in out
    assert "1개 모델 확인" not in out


def test_render_report_model_result_without_model_name_falls_back_to_basic(capsys) -> None:
    """model_name이 없는 결과는(비정상 입력) --model 체크로 표시되지 않는다.

    cli.py가 --model 체크 결과에는 항상 model_name을 병합해줄 것으로
    기대하므로 실전에서는 거의 발생하지 않지만, 이 값이 빠지면 조용히 기본
    체크로 취급되는 게(크래시보다는) 안전한 기본값이다.
    """
    raw = {k: v for k, v in _MODEL_MODE_RAW.items() if k != "model_name"}
    basic_result = judge_result(_OK_RAW)
    model_result = judge_result(raw)

    render_report([basic_result, model_result])

    out = capsys.readouterr().out
    assert out.count("기본 체크") == 2
    assert "모델 체크" not in out


def test_render_report_two_results_json_mode_unaffected(capsys) -> None:
    """--json 모드는 그룹 라벨과 무관하게 results를 가공 없이 그대로 담는다(cli.md 계약)."""
    basic_result = judge_result(_OK_RAW)
    model_result = judge_result(_MODEL_MODE_RAW)

    render_report([basic_result, model_result], json_output=True)

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert len(payload["results"]) == 2
    assert "기본 체크" not in out
    assert "모델 체크" not in out
