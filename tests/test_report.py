"""render_report() 출력 형식 검증. 출력 예시는 docs/contracts/cli.md 참고."""

from __future__ import annotations

import json

import pytest

from preflight import report
from preflight.canary.judge import judge_result
from preflight.cli import _aggregate_verdict, get_exit_code
from preflight.report import _truncate_error_log, render_report

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


def test_render_report_ok_without_memory_delta_shows_neutral_line_not_success(capsys) -> None:
    """device=cpu인데 memory_delta_mb를 못 재고도 status="ok"인 경우(#60 실측) —

    "메모리 이동 확인됨"은 측정도 안 한 값을 확인됐다고 말하는 거짓 확인이라,
    초록 ✔ 대신 중립적인 ℹ 문구여야 한다.
    """
    raw = {**_OK_RAW, "device": "cpu", "memory_delta_mb": None}
    result = judge_result(raw)

    render_report([result])

    out = capsys.readouterr().out
    assert "메모리 이동 확인됨" not in out
    assert "GPU 메모리 이동 없음" in out
    assert "ℹ" in out


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


def test_render_report_status_error_known_cause_text(capsys) -> None:
    """ModelConfigError(#62)면 표제가 "원인 미상" 대신 그 메시지를 보여준다(#83)."""
    raw = {
        **_OK_RAW,
        "status": "error",
        "error_log": (
            "Traceback (most recent call last):\n"
            '  File "worker.py", line 1, in <module>\n'
            "    raise ModelConfigError(message) from None\n"
            "preflight.canary.model.ModelConfigError: 모델을 찾을 수 없음: "
            "typo-org/no-such-model (HF Hub에 해당 저장소 없음 — 모델명 오타 또는 비공개 저장소)"
        ),
    }
    result = judge_result(raw)

    render_report([result])

    out = capsys.readouterr().out
    assert "모델을 찾을 수 없음: typo-org/no-such-model" in out
    assert "원인 미상" not in out


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


def test_render_report_quant_fallback_does_not_repeat_missing_cause(capsys) -> None:
    """bnb 미설치가 확정이면 폴백 줄은 **원인을 말하지 않는다** (#124 리뷰 ①).

    같은 조건에서 `_missing_stack_lines`가 `⚠ 4bit 사용 불가` 줄로 원인을 이미
    말한다(#117). 폴백 줄까지 "bitsandbytes가 설치되어 있지 않아"로 시작하면
    나란한 두 줄이 같은 원인을 반복해 서로 다른 두 문제로 읽힌다. 이 줄이 남겨야
    할 고유 정보는 "무엇으로 측정했는가"뿐이다.
    """
    raw = {
        **_OK_RAW,
        "device": "cuda",
        "quant_backend": "nn-linear-fallback",
        "env": {"bitsandbytes_installed": False},
    }

    render_report([judge_result(raw)])

    out = capsys.readouterr().out
    assert "nn.Linear로 대체 실행됨" in out
    assert "설치되어 있지 않아" not in out
    assert "미설치 또는 구버전" not in out
    # 원인은 ⚠ 줄에서 딱 한 번만 나온다.
    assert out.count("bitsandbytes가 없어") == 1


def test_render_report_quant_fallback_keeps_vague_message_when_unknown(capsys) -> None:
    """`None`(못 읽음)이면 단정하지 않고 기존 뭉뚱그린 문구를 쓴다 (#44)."""
    raw = {
        **_OK_RAW,
        "device": "cuda",
        "quant_backend": "nn-linear-fallback",
        "env": {"bitsandbytes_installed": None},
    }

    render_report([judge_result(raw)])

    assert "미설치 또는 구버전" in capsys.readouterr().out


def test_render_report_quant_fallback_message_does_not_assume_old_version(capsys) -> None:
    """bitsandbytes 미설치 환경도 있는데 "구버전"이라고 단정하지 않는다(#60)."""
    raw = {**_OK_RAW, "device": "cuda", "quant_backend": "nn-linear-fallback"}
    result = judge_result(raw)

    render_report([result])

    out = capsys.readouterr().out
    assert "구버전 bitsandbytes 등으로" not in out
    assert "미설치 또는 구버전" in out


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


def test_render_report_with_fix_long_command_stays_one_line(capsys, monkeypatch) -> None:
    """#91: 좁은 터미널 폭에서도 FIX 줄에 실제 개행 문자가 끼어들면 안 된다 —
    끼어들면 복사해 붙였을 때 --index-url과 값이 갈라진다."""
    monkeypatch.setenv("COLUMNS", "40")
    long_command = (
        r"C:\Users\tkddu\preflight-test\.venv\Scripts\python.exe -m pip install "
        r"--force-reinstall torch --index-url https://download.pytorch.org/whl/cu124"
    )
    raw = {**_OK_RAW, "status": "oom", "memory_delta_mb": None, "elapsed_ms": None}
    result = judge_result(raw)
    result["fix"] = {
        "cause": "oom",
        "message": "CUDA Out of Memory",
        "fix_command": long_command,
    }

    render_report([result])

    out = capsys.readouterr().out
    assert f"FIX: {long_command}" in out


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

    assert set(payload.keys()) == {"results", "summary", "notices", "exit_code_hint"}
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


def test_render_report_json_mode_warn_only_exit_hint_two(capsys) -> None:
    """FAIL 없이 WARN만 있으면 exit_code_hint는 2다 — cli.get_exit_code()와 같은 값(#70).

    이진(0/1) 계산이었던 예전 버전은 여기서 1을 냈다 — 힌트를 믿고 분기하는 CI
    스크립트가 WARN을 FAIL로 오인하는 원인이었다.
    """
    raw = {**_OK_RAW, "cpu_multiplier": 1.83}  # cpu_multiplier_low → WARN
    result = judge_result(raw)
    assert result["verdict"] == "WARN"

    render_report([result], json_output=True)

    payload = json.loads(capsys.readouterr().out)
    assert payload["exit_code_hint"] == 2
    assert payload["exit_code_hint"] == get_exit_code(_aggregate_verdict([result]))


def test_render_report_json_mode_fail_and_warn_exit_hint_one(capsys) -> None:
    """FAIL과 WARN이 섞이면 FAIL이 우선이라 exit_code_hint는 1이다."""
    warn_raw = {**_OK_RAW, "cpu_multiplier": 1.83}  # WARN
    fail_raw = {**_OK_RAW, "device": "cpu"}  # quant_layer_device_cpu → FAIL
    warn_result = judge_result(warn_raw)
    fail_result = judge_result(fail_raw)
    assert warn_result["verdict"] == "WARN"
    assert fail_result["verdict"] == "FAIL"

    render_report([warn_result, fail_result], json_output=True)

    payload = json.loads(capsys.readouterr().out)
    assert payload["exit_code_hint"] == 1
    assert payload["exit_code_hint"] == get_exit_code(
        _aggregate_verdict([warn_result, fail_result])
    )


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


def test_render_report_elapsed_seconds_under_one_shows_under_one_second(capsys) -> None:
    """0.27초처럼 반올림하면 "0초"가 되는 실행이 "안 돌았나?"로 안 읽히게 한다(#60)."""
    result = judge_result(_OK_RAW)

    render_report([result], elapsed_seconds=0.27)

    out = capsys.readouterr().out
    assert "소요 시간 1초 미만" in out
    assert "소요 시간 0초" not in out


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


def test_render_report_model_mode_shows_quant_fallback_line(capsys) -> None:
    """--model 모드에서도 4bit 폴백 사실은 화면에 나온다 (#66).

    폴백이면 VRAM 실측값이 QLoRA(4bit) 기준이 아니라 양자화 없는 베이스 기준이라 크게
    나온다 — 그 전제를 안 알려주면 사용자는 "이 GPU로는 무리"라는 정반대
    결론을 낸다. 판정 줄이 아니므로 문제 개수에는 들어가지 않는다.
    """
    result = judge_result({**_MODEL_MODE_RAW, "quant_backend": "nn-linear-fallback"})
    assert result["verdict"] == "PASS"

    render_report([result])

    out = capsys.readouterr().out
    assert "4bit 레이어 폴백" in out
    assert "양자화 없는 베이스로 실측됨" in out
    # #75에서 폴백이 전체 파인튜닝을 그만뒀으므로 옛 문구는 남아 있으면 안 된다.
    assert "fp32 전체 모델" not in out
    assert "VRAM 실측" in out
    assert "문제 없음" in out


def test_render_report_model_mode_has_no_fallback_line_when_4bit_worked(capsys) -> None:
    """정상 4bit 실행에는 폴백 줄이 없다 — 폴백일 때만 나오는 줄이다."""
    result = judge_result(_MODEL_MODE_RAW)

    render_report([result])

    assert "4bit 레이어 폴백" not in capsys.readouterr().out


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


def test_render_report_group_label_brackets_survive_rich_markup(capsys) -> None:
    """model_name의 대괄호는 rich 마크업으로 먹히지 않고 원문 그대로 표제에 남는다(#67)."""
    model_result = judge_result({**_MODEL_MODE_RAW, "model_name": r"C:\models\ckpt[v2]"})

    render_report([judge_result(_OK_RAW), model_result])

    out = capsys.readouterr().out
    assert r"모델 체크: C:\models\ckpt[v2]" in out


def test_render_report_group_label_closing_tag_does_not_crash(capsys) -> None:
    """model_name에 "[/bold]" 같은 닫는 태그 꼴이 들어가도 MarkupError 없이 끝까지 렌더된다(#67)."""
    model_result = judge_result({**_MODEL_MODE_RAW, "model_name": "weird[/bold]name"})

    render_report([judge_result(_OK_RAW), model_result])  # MarkupError가 나면 여기서 터진다

    out = capsys.readouterr().out
    assert "weird[/bold]name" in out


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


def test_render_report_json_mode_warn_with_skipped_item_exit_hint_two(capsys) -> None:
    """WARN 판정 + 생략 항목이 섞여도 exit_code_hint는 2 그대로다."""
    warn_raw = {**_OK_RAW, "cpu_multiplier": 1.83}
    warn_result = judge_result(warn_raw)
    assert warn_result["verdict"] == "WARN"

    render_report([warn_result, _SKIPPED_ITEM], json_output=True)

    payload = json.loads(capsys.readouterr().out)
    assert payload["exit_code_hint"] == 2


def test_render_report_exit_code_hint_always_matches_cli_get_exit_code(capsys) -> None:
    """report.py의 exit_code_hint는 항상 cli.get_exit_code(cli._aggregate_verdict(...))와 같다(#70 완료 조건).

    한쪽만 고치고 다른 쪽을 잊어버리는 회귀를 막기 위해, 두 모듈이 각자 계산한
    값을 여러 조합에서 직접 비교한다.
    """
    pass_result = judge_result(_OK_RAW)
    warn_result = judge_result({**_OK_RAW, "cpu_multiplier": 1.83})
    fail_result = judge_result({**_OK_RAW, "device": "cpu"})

    combinations = [
        [pass_result],
        [warn_result],
        [fail_result],
        [pass_result, warn_result],
        [pass_result, fail_result],
        [warn_result, fail_result],
        [pass_result, warn_result, fail_result],
        [pass_result, _SKIPPED_ITEM],
        [warn_result, _SKIPPED_ITEM],
        [fail_result, _SKIPPED_ITEM],
    ]

    for results in combinations:
        render_report(results, json_output=True)
        payload = json.loads(capsys.readouterr().out)
        expected = get_exit_code(_aggregate_verdict(results))
        assert payload["exit_code_hint"] == expected, results


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


def test_render_report_reverified_block_gets_label(capsys) -> None:
    """재확인 블록은 결과가 하나여도 "(재확인)" 표제를 붙인다 (#57).

    표시가 없으면 수정 전후 화면이 똑같아, 사용자는 화면의 ✔이 수정 덕분인지
    원래 그랬는지 알 수 없다.
    """
    result = {**judge_result(_OK_RAW), "reverified": True}

    render_report([result])

    out = capsys.readouterr().out
    assert "기본 체크 (재확인)" in out


def test_render_report_notices_shown_in_text(capsys) -> None:
    """--yes가 무엇을 했는지(또는 왜 아무것도 안 했는지) 화면에 나온다 (#53·#57)."""
    render_report(
        [judge_result(_OK_RAW)],
        notices=["자동 수정 실행: python -m pip install bitsandbytes"],
    )

    out = capsys.readouterr().out
    assert "자동 수정 실행: python -m pip install bitsandbytes" in out


def test_render_report_notices_included_in_json(capsys) -> None:
    """자동화 쪽도 --yes가 실제로 뭘 했는지 알아야 한다."""
    render_report([judge_result(_OK_RAW)], json_output=True, notices=["자동 수정 실패: pip"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["notices"] == ["자동 수정 실패: pip"]


def test_render_report_notices_default_to_empty_list_in_json(capsys) -> None:
    render_report([judge_result(_OK_RAW)], json_output=True)

    payload = json.loads(capsys.readouterr().out)
    assert payload["notices"] == []


def test_render_report_notice_with_brackets_is_escaped(capsys) -> None:
    """notice에 대괄호가 들어가도 rich 마크업으로 먹히지 않는다 (#67과 같은 함정)."""
    render_report([judge_result(_OK_RAW)], notices=["자동 수정 실패: pip install foo[all]"])

    out = capsys.readouterr().out
    assert "foo[all]" in out


# ── 체인된 예외의 꼬리 (#62) ──────────────────────────────────────────────────
#
# HF Hub는 없는 저장소에도 404가 아니라 401을 돌려주고, transformers가 그것을
# 3단으로 체인해 올린다. #43의 "첫 프레임 + … + 마지막 줄" 규칙은 단일 트레이스백을
# 전제해서, 최종 예외 뒤에 붙는 부연 설명("private repository … token")이 화면을
# 차지하고 정작 정확한 진단 줄이 밀려났다 — 오타를 낸 사용자가 토큰을 찾으러 갔다.

_CHAINED_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "site-packages/huggingface_hub/utils/_http.py", line 409, in hf_raise_for_status\n'
    "    response.raise_for_status()\n"
    "httpx.HTTPStatusError: Client error '401 Unauthorized' for url "
    "'https://huggingface.co/this-org-does-not-exist/definitely-not-a-model/"
    "resolve/main/config.json'\n"
    "\n"
    "The above exception was the direct cause of the following exception:\n"
    "\n"
    "Traceback (most recent call last):\n"
    '  File "site-packages/transformers/utils/hub.py", line 470, in cached_file\n'
    "    resolved_file = hf_hub_download(path_or_repo_id, filename)\n"
    "huggingface_hub.errors.RepositoryNotFoundError: 401 Client Error. "
    "(Request ID: Root=1-68a7c0f1)\n"
    "\n"
    "Repository Not Found for url: "
    "https://huggingface.co/this-org-does-not-exist/definitely-not-a-model/"
    "resolve/main/config.json.\n"
    "Please make sure you specified the correct `repo_id` and `repo_type`.\n"
    "\n"
    "The above exception was the direct cause of the following exception:\n"
    "\n"
    "Traceback (most recent call last):\n"
    '  File "site-packages/transformers/models/auto/configuration_auto.py", line 1108, '
    "in from_pretrained\n"
    "    config_dict, unused_kwargs = PretrainedConfig.get_config_dict(...)\n"
    "OSError: this-org-does-not-exist/definitely-not-a-model is not a local folder and is "
    "not a valid model identifier listed on 'https://huggingface.co/models'\n"
    "If this is a private repository, make sure to pass a token having permission to this "
    "repo either by logging in with `hf auth login` or by passing `token=<your_token>`"
)


def test_render_report_chained_exception_keeps_final_exception_line(capsys) -> None:
    """체인된 예외에서는 마지막 줄이 아니라 마지막 **예외 줄**이 꼬리의 기준이다(#62)."""
    raw = {**_OK_RAW, "status": "error", "error_log": _CHAINED_TRACEBACK}

    render_report([judge_result(raw)])

    out = capsys.readouterr().out.replace("\n", "")
    assert "not a valid model identifier" in out


def test_render_report_chained_exception_hides_misleading_token_hint(capsys) -> None:
    """모델명 오타인데 401·token이 보이면 사용자가 인증 문제로 오해한다(#62)."""
    raw = {**_OK_RAW, "status": "error", "error_log": _CHAINED_TRACEBACK}

    render_report([judge_result(raw)])

    out = capsys.readouterr().out.replace("\n", "")
    assert "token=<your_token>" not in out
    assert "401" not in out


def test_render_report_single_traceback_tail_is_unchanged(capsys) -> None:
    """단일 트레이스백에서는 마지막 줄이 곧 마지막 예외 줄이라 기존 동작 그대로다."""
    raw = {**_OK_RAW, "status": "import_crash", "error_log": _LONG_TRACEBACK}

    render_report([judge_result(raw)])

    out = capsys.readouterr().out.replace("\n", "")
    assert "ModuleNotFoundError: No module named 'torch'" in out
    assert "line 77, in main" in out


def test_truncate_error_log_keeps_exception_head_when_line_over_budget() -> None:
    """예외 줄 하나가 예산보다 길면 **앞을 남기고** 자른다 (#80 → #71로 계약 변경).

    원래는 "앞을 자르면 타입만 남고 메시지가 사라진다"는 이유로 그 줄을 통째로
    남겼는데(#80), 상한이 없어 절삭이 무력화됐다 — 입력 5070자가 출력 5076자가
    되어 화면이 로그에 뒤덮였다(#71). 앞을 남기면 **타입과 메시지 앞부분이 둘 다**
    보존되므로 원래 의도는 그대로 지켜진다.

    체인 예외의 부연 설명("private repository … token")을 버리는 #62의 성질도
    함께 유지돼야 한다.
    """
    long_message = "x" * 400
    log = (
        "Traceback (most recent call last):\n"
        '  File "a.py", line 1, in <module>\n'
        "    boom()\n"
        f"OSError: {long_message}\n"
        "If this is a private repository, make sure to pass a token"
    )

    result = _truncate_error_log(log)

    assert "OSError: xxx" in result
    assert "private repository" not in result
    assert len(_log_body(result)) <= 200


def test_render_report_long_path_first_frame_keeps_file_and_line(capsys) -> None:
    """첫 프레임 줄이 길면 **뒤**(파일명·줄 번호·함수명)를 남긴다(#56).

    실제 설치 경로(venv 안 site-packages)는 거의 항상 80자를 넘는다. 앞을 남기면
    사용자 홈 경로 조각만 남고 "어느 파일 몇 번째 줄에서 죽었는지"가 사라졌다
    (QA 실측: `File "C:/Users/.../C--Users-bells-OneDrive---------…`).
    """
    log = (
        "Traceback (most recent call last):\n"
        '  File "/home/someone/very/long/virtualenv/path/that/keeps/going/lib/python3.11/'
        'site-packages/preflight/canary/worker.py", line 77, in main\n'
        "    torch = _import_canary_stack()\n"
        "ModuleNotFoundError: No module named 'torch'"
    )
    raw = {**_OK_RAW, "status": "import_crash", "error_log": log}

    render_report([judge_result(raw)])

    out = capsys.readouterr().out.replace("\n", "")
    assert 'worker.py", line 77, in main' in out
    assert "ModuleNotFoundError: No module named 'torch'" in out


# --- 없는 라이브러리마다 한 줄 (#117) ---


def _stack_result(env: dict) -> dict:
    return {
        "status": "ok",
        "device": "cuda",
        "verdict": "WARN",
        "reasons": ["qlora_stack_not_installed"],
        "quant_backend": "nn-linear-fallback",
        "env": env,
    }


def test_missing_stack_draws_one_line_per_library() -> None:
    """원인도 결과도 다르므로 뭉치지 않는다 — 4bit 불가 vs LoRA 불가 (#117)."""
    lines = report._missing_stack_lines(
        _stack_result({"bitsandbytes_installed": False, "peft_installed": False})
    )

    assert len(lines) == 2
    assert any("bitsandbytes" in line.text for line in lines)
    assert any("peft" in line.text for line in lines)
    # 정보성 표시가 아니라 문제로 센다 — 이 상태로는 학습이 안 된다.
    assert all(line.is_problem for line in lines)


def test_missing_stack_says_what_happens_to_the_user() -> None:
    """ "우리 진단이 폴백했다"가 아니라 "당신 학습이 죽는다"를 말한다 (#117)."""
    (line,) = report._missing_stack_lines(_stack_result({"bitsandbytes_installed": False}))

    assert "ImportError" in (line.detail or ""), line


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"bitsandbytes_installed": None, "peft_installed": None},
        {"bitsandbytes_installed": True, "peft_installed": True},
    ],
)
def test_no_missing_stack_line_when_not_confirmed(env: dict) -> None:
    assert report._missing_stack_lines(_stack_result(env)) == []


def test_missing_stack_lines_are_not_repeated_in_model_mode(capsys) -> None:
    """`--model` 모드에서 같은 경고가 두 번 찍히지 않는다 (#117).

    설치 여부는 모델과 무관한 환경 사실이라 기본 체크와 모델 체크의 `env`가 항상
    같다. `_build_lines`의 모드 분기 **밖**에서 그리면 두 블록에 똑같은 줄이 붙고,
    문제 수까지 실제의 두 배로 부풀려진다.
    """
    env = {"bitsandbytes_installed": False, "peft_installed": False}
    basic = _stack_result(env)
    model = {**_stack_result(env), "model_name": "gpt2", "memory_delta_mb": 100.0}

    render_report([basic, model])
    out = capsys.readouterr().out

    assert out.count("4bit 사용 불가") == 1, out
    assert out.count("LoRA 사용 불가") == 1, out
    assert "2개 문제 발견" in out, out


# ── 절삭 예산 상한 (#71) ──────────────────────────────────────────────────────
#
# 마지막 예외 줄이 예산보다 길면 "그 줄만이라도 통째로" 남기는 분기에 상한이 없어,
# 절삭이 통째로 무력화되고 원문보다 긴 출력이 나왔다(입력 5070자 → 출력 5076자).


def _log_body(rendered: str) -> str:
    """절삭 결과에서 안내 문구를 뺀 로그 본문. 예산은 본문에만 적용된다."""
    from preflight.report import _TRUNCATION_NOTE

    return rendered.rsplit("\n" + _TRUNCATION_NOTE, 1)[0]


def test_truncate_error_log_caps_long_final_exception_line() -> None:
    """마지막 예외 줄이 아무리 길어도 본문이 예산을 넘지 않는다(#71)."""
    log = (
        "Traceback (most recent call last):\n"
        + '  File "hub.py", line 100, in _get\n' * 20
        + "RuntimeError: "
        + "X" * 5000
    )

    result = _truncate_error_log(log)

    assert len(_log_body(result)) <= 200
    assert len(result) < len(log)


def test_truncate_error_log_keeps_exception_type_when_clipping() -> None:
    """길어서 자르더라도 예외 타입과 메시지 앞부분은 남는다 — 앞이 정보다(#71)."""
    log = (
        "Traceback (most recent call last):\n"
        '  File "a.py", line 1, in <module>\n'
        "    boom()\n"
        "RuntimeError: 실제 원인 메시지가 여기 " + "길게 " * 200
    )

    result = _truncate_error_log(log)

    assert "RuntimeError: 실제 원인 메시지가 여기" in result
    assert len(_log_body(result)) <= 200


def test_truncate_error_log_caps_long_single_line() -> None:
    """개행 없는 한 줄 로그도 예산을 지킨다(#71) — engine이 만드는 형태다."""
    log = "OSError: " + "y" * 3000

    result = _truncate_error_log(log)

    assert len(_log_body(result)) <= 200
    assert result.startswith("OSError: ")


def test_render_report_notice_long_command_stays_one_line(capsys, monkeypatch) -> None:
    """`안내:` 메시지에도 실제 개행이 끼면 안 된다 — 본문에 pip 명령이 박혀 있다.

    #91이 `FIX:` 줄에 soft_wrap을 넣었는데 바로 아래 `안내:` 분기는 놓쳤다.
    `fix_command`가 없는 cause가 훨씬 많아(_FIX_MAP 대부분) 실제로는 이쪽이 더
    자주 나가고, `torch_cpu_only_build_no_gpu`처럼 메시지 안에 실행 가능한 명령이
    들어 있는 것들이 있다. 개행이 끼면 복사한 명령이 쪼개져 실패한다(QA 3차 실측).
    """
    monkeypatch.setenv("COLUMNS", "40")
    message = (
        "설치된 torch가 CPU 전용 빌드이고 NVIDIA GPU도 조회되지 않았다 — "
        "pip install --force-reinstall torch "
        "--index-url https://download.pytorch.org/whl/cu124"
    )
    raw = {**_OK_RAW, "status": "oom", "memory_delta_mb": None, "elapsed_ms": None}
    result = judge_result(raw)
    result["fix"] = {
        "cause": "torch_cpu_only_build_no_gpu",
        "message": message,
        "fix_command": None,
    }

    render_report([result])

    out = capsys.readouterr().out
    assert f"안내: {message}" in out


def test_render_report_model_name_newline_does_not_inject_line(capsys) -> None:
    """model_name의 개행이 표제 아래에 가짜 줄로 들어가면 안 된다 (#127).

    `--model` 값은 사용자 입력이라 스크립트로 조립하면 개행이 섞일 수 있다.
    escape()(#67)는 rich 마크업만 막고 개행은 막지 않아, 판정 줄처럼 보이는
    임의 텍스트를 화면에 끼워 넣을 수 있었다.
    """
    injected = "foo/bar\nFAKE: 모든 항목 정상"
    model_result = judge_result({**_MODEL_MODE_RAW, "model_name": injected})

    render_report([judge_result(_OK_RAW), model_result])

    out = capsys.readouterr().out
    assert "모델 체크: foo/bar FAKE: 모든 항목 정상" in out
    assert "\nFAKE:" not in out


def test_render_report_cpu_verdict_does_not_name_a_layer_that_was_never_built(capsys) -> None:
    """bnb가 없어 4bit 레이어가 만들어지지도 않았으면 그 이름으로 판정하지 않는다.

    judge.py는 `quant_backend`와 무관하게 `device == "cpu"`면 FAIL을 매긴다 —
    bitsandbytes조차 없는 환경이 cpu인 채로 PASS로 새던 구멍을 막은 올바른
    규칙이다(#18). 그런데 화면 문구만 "bitsandbytes 4bit 레이어"로 고정돼 있어
    **있지도 않은 레이어를 검사해 cpu로 판정했다**는 말이 나갔다(#124 리뷰 ①).
    """
    raw = {
        **_OK_RAW,
        "device": "cpu",
        "quant_backend": "nn-linear-fallback",
        "env": {"bitsandbytes_installed": False},
    }

    render_report([judge_result(raw)])

    out = capsys.readouterr().out
    assert "연산 레이어" in out
    assert "GPU를 타지 못했다" in out
    assert "bitsandbytes 4bit 레이어" not in out


def test_render_report_cpu_verdict_keeps_4bit_wording_when_layer_was_built(capsys) -> None:
    """4bit 레이어가 실제로 만들어졌는데 cpu로 떨어진 경우는 문구가 그대로다.

    이쪽은 "조용한 CPU 폴백"이 정확한 설명이다 — bitsandbytes가 있고 레이어도
    만들어졌는데 연산이 GPU를 타지 않은, 원래 이 줄이 잡으려던 상황이다.
    """
    raw = {**_OK_RAW, "device": "cpu"}

    render_report([judge_result(raw)])

    out = capsys.readouterr().out
    assert "bitsandbytes 4bit 레이어" in out
    assert "조용한 CPU 폴백" in out
    assert "연산 레이어" not in out


def test_render_report_reverified_shows_before_after(capsys) -> None:
    """재확인 블록은 "수정 전 → 수정 후"를 한 줄로 보여준다 (#88).

    재확인 결과는 1차 결과와 **교체**되므로, 재확인 대상이 하나뿐이면 화면에
    비교할 상대가 없다. `--yes`가 파는 것은 "✖였던 게 ✔로 바뀐다"인데 그 변화가
    정작 화면에 없었다 — 표제의 "(재확인)" 말고는 --yes를 쓴 화면과 안 쓴 화면이
    구별되지 않았다.
    """
    result = {**judge_result(_OK_RAW), "reverified": True, "previous_verdict": "FAIL"}

    render_report([result])

    out = capsys.readouterr().out
    assert "수정 전" in out
    assert "FAIL" in out
    assert "수정 후" in out
    assert "PASS" in out


def test_render_report_reverified_says_so_when_nothing_changed(capsys) -> None:
    """판정이 그대로면 "(변화 없음)"이라고 못박는다 (#88).

    화살표만 있으면 같은 값이 두 번 찍힌 것처럼 읽힌다. 사용자가 가장 알고 싶은
    것은 "그래서 고쳐졌나"라 명시적으로 말해준다.
    """
    raw = {**_OK_RAW, "device": "cpu"}
    result = {**judge_result(raw), "reverified": True, "previous_verdict": "FAIL"}

    render_report([result])

    assert "(변화 없음)" in capsys.readouterr().out


def test_render_report_reverified_omits_recheck_instruction(capsys) -> None:
    """재확인한 블록에는 `재확인: preflight check --yes` 안내가 안 나온다 (#88).

    방금 --yes로 재확인을 마친 화면이다. 그대로 따르면 같은 설치를 100초 넘게
    들여 반복하고 똑같은 화면을 본다(상영님 실측).
    """
    raw = {**_OK_RAW, "device": "cpu"}
    result = {
        **judge_result(raw),
        "reverified": True,
        "previous_verdict": "FAIL",
        "fix": {"cause": "x", "message": "m", "fix_command": "pip install bitsandbytes"},
    }

    render_report([result])

    out = capsys.readouterr().out
    assert "FIX: pip install bitsandbytes" in out
    assert "재확인: preflight check --yes" not in out


def test_render_report_non_reverified_keeps_recheck_instruction(capsys) -> None:
    """재확인하지 않은 블록에는 안내가 그대로 남는다 — 회귀 방지."""
    raw = {**_OK_RAW, "device": "cpu"}
    result = {
        **judge_result(raw),
        "fix": {"cause": "x", "message": "m", "fix_command": "pip install bitsandbytes"},
    }

    render_report([result])

    assert "재확인: preflight check --yes" in capsys.readouterr().out


def test_render_report_reverified_delta_line_does_not_change_counts(capsys) -> None:
    """수정 전/후 줄은 항목 수에도 문제 수에도 안 들어간다 (#88 완료 조건).

    판정을 받은 항목이 아니라 그 항목에 대한 설명이다. 여기가 어긋나면 화면의
    "N개 항목 확인"과 --json의 `summary.total_items`가 갈라진다.
    """
    base = judge_result(_OK_RAW)
    with_delta = {**base, "reverified": True, "previous_verdict": "FAIL"}

    render_report([base])
    plain = capsys.readouterr().out
    render_report([with_delta])
    reverified = capsys.readouterr().out

    def summary(text: str) -> str:
        return next(line for line in text.splitlines() if "항목 확인" in line)

    assert "수정 전" in reverified
    assert summary(plain) == summary(reverified)


def test_render_report_reverified_without_previous_verdict_omits_line(capsys) -> None:
    """1차 판정이 안 실려 오면 줄을 만들지 않는다 (#88).

    `render_report`를 직접 부르는 호출자는 `previous_verdict`를 모를 수 있다.
    비교할 것이 없으면 지금까지의 화면 그대로 둔다.
    """
    result = {**judge_result(_OK_RAW), "reverified": True}

    render_report([result])

    assert "수정 전" not in capsys.readouterr().out


def test_render_report_reverified_delta_survives_failed_status(capsys) -> None:
    """status가 ok가 아니어도 수정 전/후 줄은 나온다 (#88).

    `_build_check_lines`는 그 경우 판정 줄 하나만 내고 일찍 끝나는데, **수정 후에도
    여전히 실패한 경우가 바로 그 경로다** — "고쳐지지 않았다"를 알려줘야 하는 때다.
    """
    raw = {**_OK_RAW, "status": "import_crash", "error_log": "boom"}
    result = {**judge_result(raw), "reverified": True, "previous_verdict": "FAIL"}

    render_report([result])

    out = capsys.readouterr().out
    assert "수정 전" in out
    assert "(변화 없음)" in out


def test_render_report_notice_command_stays_one_line(capsys, monkeypatch) -> None:
    """알림에 실린 명령에도 실제 개행이 끼면 안 된다 (#149).

    `자동 수정 실행: <command>`처럼 알림에는 거의 항상 명령이 들어간다. rich 기본값은
    폭에 맞춰 문자열 자체에 개행을 끼워 넣어, 복사하면 --index-url과 값이 분리된다.
    #91(FIX:)·#128(안내:)과 같은 버그의 세 번째 경로였다.
    """
    monkeypatch.setenv("COLUMNS", "40")
    notice = (
        "자동 수정 실행: C:/venv/Scripts/python.exe -m pip install --force-reinstall "
        "torch --index-url https://download.pytorch.org/whl/cu130"
    )

    render_report([judge_result(_OK_RAW)], notices=[notice])

    assert notice in capsys.readouterr().out


def test_render_report_fallback_shows_reason(capsys) -> None:
    """폴백 줄이 **왜 폴백했는지**를 detail로 보여준다 (#147).

    예전에는 `_build_qlora_model`의 `except Exception`이 예외를 통째로 삼켜서,
    우리 코드의 버그(#134)까지 화면에는 "4bit 레이어 구성 실패 → 폴백"으로만
    보였다. 읽는 사람은 "이 환경이 4bit을 못 쓰는구나"로 받아들일 뿐, 도구 자신의
    결함을 의심할 단서가 없었다.
    """
    raw = {
        **_OK_RAW,
        "quant_backend": "nn-linear-fallback",
        "quant_fallback_reason": "NotImplementedError: Cannot copy out of meta tensor; no data!",
        "env": {"bitsandbytes_installed": True},
    }

    render_report([judge_result(raw)])

    out = capsys.readouterr().out
    assert "4bit 레이어 폴백" in out
    assert "Cannot copy out of meta tensor" in out


def test_render_report_fallback_reason_hidden_when_bnb_known_missing(capsys) -> None:
    """bnb 미설치가 확정인 기본 체크에서는 사유를 붙이지 않는다 (#147).

    같은 화면의 `⚠ 4bit 사용 불가` 줄이 이미 원인을 말한다(#117). 여기에
    `ModuleNotFoundError`까지 붙이면 #142에서 막 없앤 원인 중복이 형태만 바꿔
    되살아난다.
    """
    raw = {
        **_OK_RAW,
        "quant_backend": "nn-linear-fallback",
        "quant_fallback_reason": "ModuleNotFoundError: No module named 'bitsandbytes'",
        "env": {"bitsandbytes_installed": False},
    }

    render_report([judge_result(raw)])

    out = capsys.readouterr().out
    assert "4bit 레이어 폴백" in out
    assert "ModuleNotFoundError" not in out
    # 원인은 ⚠ 줄에서 딱 한 번만 나온다.
    assert out.count("bitsandbytes가 없어") == 1


def test_render_report_model_mode_fallback_shows_reason_even_if_bnb_missing(capsys) -> None:
    """--model 블록에는 ⚠ 줄이 없으므로 사유를 항상 보여준다 (#147).

    설치 여부 줄은 기본 체크 한 곳에서만 그린다(#117). 모델 블록에서 사유까지
    감추면 아무도 말해주지 않는 상태가 된다 — #134가 숨어 있던 자리가 거기다.
    """
    raw = {
        **_MODEL_MODE_RAW,
        "model_name": "meta-llama/Llama-3.1-8B",
        "quant_backend": "nn-linear-fallback",
        "quant_fallback_reason": "ModuleNotFoundError: No module named 'bitsandbytes'",
        "env": {"bitsandbytes_installed": False},
    }

    render_report([judge_result(raw)])

    assert "ModuleNotFoundError" in capsys.readouterr().out


def test_render_report_fallback_without_reason_adds_no_line(capsys) -> None:
    """사유가 없으면(구버전 페이로드) 지금까지의 화면 그대로다 (#147)."""
    raw = {**_OK_RAW, "quant_backend": "nn-linear-fallback", "env": {}}

    render_report([judge_result(raw)])
    without = capsys.readouterr().out
    render_report([judge_result({**raw, "quant_fallback_reason": "RuntimeError: boom"})])
    with_reason = capsys.readouterr().out

    assert "4bit 레이어 폴백" in without
    assert "RuntimeError: boom" not in without
    assert "RuntimeError: boom" in with_reason
    # 사유 줄 하나만 늘어난다 — 다른 줄 구성은 건드리지 않는다.
    assert len(with_reason.splitlines()) == len(without.splitlines()) + 1


def test_render_report_fallback_reason_is_capped(capsys) -> None:
    """긴 폴백 사유는 화면에서 잘린다 (#147, 상영님 리뷰).

    detail 줄은 `overflow="fold"`라 잘리지 않고 통째로 접힌다 — HF Hub의 401 체인이나
    bitsandbytes의 CUDA 진단 블록이 그대로 오면 화면을 덮는다. `error_log`가 같은
    `"<예외 종류>: <메시지>"` 형식인데 200자 예산을 지키는 것과 어긋났다.
    `--json`의 원본은 그대로다 — 자르는 것은 화면뿐이다.
    """
    reason = "RuntimeError: " + "가" * 500
    raw = {
        **_OK_RAW,
        "quant_backend": "nn-linear-fallback",
        "quant_fallback_reason": reason,
        "env": {"bitsandbytes_installed": True},
    }

    render_report([judge_result(raw)])

    out = capsys.readouterr().out
    assert "RuntimeError" in out
    assert reason not in out
    assert "…" in out
