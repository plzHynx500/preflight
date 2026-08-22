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


# ── 생략된 체크(skipped) — cli.md "결과 집계" / #36 ──────────────────────────
#
# fail-fast로 모델 체크가 생략되면 CLI가 넘기는 항목에는 status·verdict·reasons가
# 통째로 없다. 판정 줄을 그리려 들면 status가 None이라 "status=None 감지" 같은
# 빨간 ✖ 줄이 나가고 문제 개수까지 하나 늘어났다(#36).

_SKIPPED_ITEM = {
    "model_name": "meta-llama/Llama-3.1-8B",
    "batch_size": 2,
    "seq_len": 2048,
    "skipped": "환경 체크 실패",
}

_FAILED_BASIC = {
    **_OK_RAW,
    "device": "cpu",  # quant_layer_device_cpu → FAIL
}


def test_render_report_skipped_item_shows_reason_not_failure_line(capsys) -> None:
    """생략 항목은 생략 사유 한 줄만 그린다 — 판정 줄(✖)이 아니다."""
    basic = judge_result(_FAILED_BASIC)
    assert basic["verdict"] == "FAIL"

    render_report([basic, _SKIPPED_ITEM])

    out = capsys.readouterr().out
    assert "— 환경 체크 실패로 생략" in out
    # 생략 항목이 별개의 실패처럼 보이면 안 된다.
    assert "status=None" not in out
    assert "정상 실행 불가" not in out


def test_render_report_skipped_item_not_counted_as_item_or_problem(capsys) -> None:
    """생략 항목은 판정된 적이 없으므로 항목 수·문제 수 어디에도 안 들어간다.

    cli.md의 생략 예시가 "3개 항목 확인 · 1개 문제 발견"인데, 고치기 전에는
    "4개 항목 확인 · 2개 문제 발견"이 나왔다(#36).
    """
    basic = judge_result(_FAILED_BASIC)

    render_report([basic, _SKIPPED_ITEM])

    out = capsys.readouterr().out
    assert "3개 항목 확인" in out
    assert "1개 문제 발견" in out


def test_render_report_skipped_item_keeps_group_label(capsys) -> None:
    """생략됐어도 model_name이 있으므로 표제는 정상적으로 "모델 체크: <name>"이다."""
    basic = judge_result(_FAILED_BASIC)

    render_report([basic, _SKIPPED_ITEM])

    out = capsys.readouterr().out
    assert "기본 체크" in out
    assert "모델 체크: meta-llama/Llama-3.1-8B" in out


def test_render_report_skipped_item_json_summary_excludes_it(capsys) -> None:
    """--json의 summary 집계에서도 생략 항목은 빠진다. results 배열에는 그대로 남는다."""
    basic = judge_result(_FAILED_BASIC)

    render_report([basic, _SKIPPED_ITEM], json_output=True)

    payload = json.loads(capsys.readouterr().out)

    # 원본은 가공 없이 그대로 통과된다(cli.md 계약).
    assert len(payload["results"]) == 2
    assert payload["results"][1]["skipped"] == "환경 체크 실패"
    # 집계에는 안 들어간다.
    assert payload["summary"]["total_items"] == 3
    assert payload["summary"]["fail"] == 1


def test_render_report_skipped_item_does_not_break_exit_code_hint(capsys) -> None:
    """verdict가 없는 생략 항목 때문에 exit_code_hint가 왜곡되지 않는다.

    생략은 기본 체크 FAIL일 때만 일어나 결과가 뒤집히진 않지만, 판정 대상에서
    명시적으로 빼서 "우연히 맞는" 상태를 남기지 않는다.
    """
    all_pass = judge_result(_OK_RAW)
    assert all_pass["verdict"] == "PASS"

    # 판정된 항목이 전부 PASS면, 생략 항목이 섞여 있어도 힌트는 0이어야 한다.
    render_report([all_pass, _SKIPPED_ITEM], json_output=True)

    payload = json.loads(capsys.readouterr().out)
    assert payload["exit_code_hint"] == 0


def test_render_report_fix_block_comes_after_all_check_blocks(capsys) -> None:
    """FIX는 체크 블록 사이가 아니라 전부 그린 뒤 한 번에 나온다(cli.md 출력 예시)."""
    basic = judge_result(_FAILED_BASIC)
    basic["fix"] = {
        "cause": "bnb_not_compiled_with_cuda",
        "message": "bitsandbytes가 CUDA 지원 없이 빌드됨",
        "fix_command": "pip install bitsandbytes --upgrade --force-reinstall",
    }

    render_report([basic, _SKIPPED_ITEM])

    out = capsys.readouterr().out
    assert out.index("모델 체크: meta-llama/Llama-3.1-8B") < out.index("FIX:")
    assert out.index("— 환경 체크 실패로 생략") < out.index("FIX:")


# ── error_log 표시 (#43) ──────────────────────────────────────────────────────
#
# 파이썬 트레이스백은 진짜 원인(예외 타입·메시지)이 마지막 줄에 온다. 앞에서
# 200자만 남기고 자르던 예전 방식은 그 줄을 항상 잘라내 파일 경로만 남겼다.

_LONG_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "/home/someone/preflight/src/preflight/canary/worker.py", line 77, in main\n'
    "    torch = _import_canary_stack()\n"
    '  File "/home/someone/preflight/src/preflight/canary/worker.py", line 61, '
    "in _import_canary_stack\n"
    "    import torch\n"
    "ModuleNotFoundError: No module named 'torch'"
)
assert len(_LONG_TRACEBACK) > 200, "이 픽스처는 잘림 경로를 타야 의미가 있다"


def test_render_report_long_error_log_keeps_last_line(capsys) -> None:
    """잘리더라도 트레이스백의 마지막 줄(실제 예외)은 항상 화면에 남는다(#43)."""
    raw = {**_OK_RAW, "status": "import_crash", "error_log": _LONG_TRACEBACK}

    render_report([judge_result(raw)])

    out = capsys.readouterr().out
    assert "ModuleNotFoundError: No module named 'torch'" in out
    assert "…" in out


def test_render_report_long_error_log_keeps_first_frame(capsys) -> None:
    """단순 tail이 아니라 "무엇을 하다가 죽었는지"(첫 프레임)도 함께 남긴다."""
    raw = {**_OK_RAW, "status": "import_crash", "error_log": _LONG_TRACEBACK}

    render_report([judge_result(raw)])

    out = capsys.readouterr().out.replace("\n", "")  # rich가 폭에 맞춰 줄바꿈할 수 있다
    assert "line 77, in main" in out
    # 정보 없는 헤더 줄은 첫 프레임으로 건너뛴다.
    assert "most recent call last" not in out


def test_render_report_error_log_at_limit_is_untouched(capsys) -> None:
    """경계값: 정확히 200자면 자르지 않고 전문이 나온다."""
    log = "E" * 200
    raw = {**_OK_RAW, "status": "error", "error_log": log}

    render_report([judge_result(raw)])

    out = capsys.readouterr().out.replace("\n", "")  # rich가 폭에 맞춰 줄바꿈할 수 있다
    assert log in out
    assert "…" not in out


def test_render_report_error_log_brackets_survive_rich_markup(capsys) -> None:
    """error_log의 대괄호는 rich 마크업으로 먹히지 않는다.

    engine이 크래시 로그에 "[stdout]"/"[stderr]" 라벨을 넣는데, 이스케이프 없이
    rich에 넘기면 라벨이 조용히 사라지고(실측) "[/x]" 꼴은 MarkupError로 죽는다.
    """
    raw = {**_OK_RAW, "status": "error", "error_log": "[stderr]\nCUDA Setup failed [/bnb]"}

    render_report([judge_result(raw)])  # MarkupError가 나면 여기서 터진다

    out = capsys.readouterr().out
    assert "[stderr]" in out
    assert "CUDA Setup failed" in out


# ── VRAM 실측값 단위 (#45) ────────────────────────────────────────────────────


def test_render_report_vram_below_1gb_shown_in_mb(capsys) -> None:
    """1GB 미만 실측값은 MB로 — `0.0GB`로 뭉개져 측정 실패처럼 보이지 않게(#45)."""
    raw = {
        **_MODEL_MODE_RAW,
        "memory_delta_mb": 18.018,
        "env": {"gpu_free_mb": 9420.0, "gpu_total_mb": 12282.0},
    }

    render_report([judge_result(raw)])

    out = capsys.readouterr().out
    assert "18MB / 9.2GB 가용 (총 12GB)" in out
    assert "0.0GB" not in out


def test_render_report_vram_unit_boundaries(capsys) -> None:
    """경계값: 1GB 직전은 MB, 1GB 정각부터 GB, 1MB 미만은 <1MB. 1GB 이상은 기존 그대로."""
    cases = [
        (972.8, "973MB"),  # 0.95GB
        (1024.0, "1.0GB"),
        (0.3, "<1MB"),
        (8600.0, "8.4GB"),  # 기존 동작 유지
    ]
    for memory_delta_mb, expected in cases:
        raw = {**_MODEL_MODE_RAW, "memory_delta_mb": memory_delta_mb}

        render_report([judge_result(raw)])

        out = capsys.readouterr().out
        assert f"{expected} 사용" in out, (memory_delta_mb, out)


def test_render_report_single_line_error_log_keeps_both_ends(capsys) -> None:
    """개행 없는 한 줄 로그는 앞뒤를 반씩 남긴다 — 출력이 입력보다 길어지지 않는다.

    engine이 f"{type(exc).__name__}: {exc}"로 만드는 한 줄 로그는 예외 이름이
    맨 앞에 온다. 첫 프레임/마지막 줄 논리를 그대로 태우면 같은 줄이 head와
    tail에 중복돼 537자 입력이 619자 출력이 됐다(PR #48 리뷰, 상영님 실측).
    """
    log = "OSError: " + "x" * 520 + " END"
    raw = {**_OK_RAW, "status": "error", "error_log": log}

    render_report([judge_result(raw)])

    out = capsys.readouterr().out.replace("\n", "")
    assert "OSError:" in out
    assert "END" in out
    assert "…" in out
    assert out.count("x") < 520


def test_render_report_truncated_error_log_says_so(capsys) -> None:
    """줄였으면 줄였다고 알린다 — 사용자가 잘린 것을 전부라고 믿지 않게(PR #48 리뷰)."""
    raw = {**_OK_RAW, "status": "import_crash", "error_log": _LONG_TRACEBACK}

    render_report([judge_result(raw)])

    out = capsys.readouterr().out.replace("\n", "")
    assert "일부만" in out
    assert "--json)" in out


def test_render_report_short_error_log_has_no_truncation_note(capsys) -> None:
    raw = {**_OK_RAW, "status": "error", "error_log": "config 조회 실패"}

    render_report([judge_result(raw)])

    out = capsys.readouterr().out
    assert "일부만" not in out
