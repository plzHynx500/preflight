"""rich 기반 CLI 리포트 출력 및 --json 직렬화. 출력 예시는 docs/contracts/cli.md 참고."""

from __future__ import annotations

import json

from rich.console import Console
from rich.markup import escape

# reason 코드(judge.py의 실제 문자열, 원문 그대로) -> 사람이 읽는 한국어 메시지.
# causes.py의 cause 코드(예: "bnb_not_compiled_with_cuda")와는 별개의 키 공간이다 —
# reason은 "어떤 조건이 걸렸는지"를, cause는 "왜"(FIX 문구용)를 나타낸다.
_REASON_MESSAGES: dict[str, str] = {
    "status_oom": "CUDA Out of Memory — 목표 batch/seq 크기로 실행 불가",
    "status_import_crash": "PyTorch/CUDA 또는 라이브러리 import 중 크래시 발생",
    "status_error": "원인 미상 오류 (모델명 오타 등 config 조회 실패 가능) — 상세 로그 확인 필요",
    "quant_layer_device_cpu": "4bit 양자화 레이어가 device=cpu로 감지됨 → 조용한 CPU 폴백",
    "quant_fallback": (
        "구버전 bitsandbytes 등으로 4bit 레이어 구성 실패 → nn.Linear로 대체 실행됨"
        " (device=cpu 판정 생략)"
    ),
    "memory_delta_high": "canary 실행만으로 가용 VRAM의 90% 이상을 소모 — 실제 학습 시 OOM 위험 높음",
    "cpu_multiplier_low": "CPU 대비 실행 속도가 2배 미만 (성능 저하 가능성)",
}

# 이 reason은 verdict(FAIL/WARN)와 무관한 정보성 표시라 문제 카운트에서 제외한다.
_INFO_ONLY_REASONS = {"quant_fallback"}

_ERROR_LOG_MAX_CHARS = 200
_TRUNCATION_NOTE = "(로그 일부만 표시 — 전문은 preflight check --json)"


class _Line:
    """화면에 찍히는 항목 한 줄 — symbol/스타일 + 문제 카운트 여부.

    `counts_as_item`은 "판정을 받은 항목인가"를 뜻한다. 생략된 체크(`skipped`)는
    화면에는 한 줄 나오지만 판정된 적이 없으므로 "N개 항목 확인"의 N에도,
    `summary.total_items`에도 들어가면 안 된다(cli.md "결과 집계").
    """

    def __init__(
        self,
        symbol: str,
        style: str,
        text: str,
        detail: str | None = None,
        is_problem: bool = False,
        counts_as_item: bool = True,
    ) -> None:
        self.symbol = symbol
        self.style = style
        self.text = text
        self.detail = detail
        self.is_problem = is_problem
        self.counts_as_item = counts_as_item


def _is_skipped(result: dict) -> bool:
    """fail-fast로 아예 실행되지 않은 체크인가(cli.md "결과 집계").

    기본 체크가 FAIL이면 모델 체크는 돌리지 않고 CLI가 `{"model_name": ...,
    "skipped": "<사유>"}` 형태의 항목을 넘긴다 — `status`·`verdict`·`reasons`가
    통째로 없다. 판정 줄을 그리려 들면 `status`가 None이라 "status=None 감지"
    같은 빨간 ✖ 줄이 나가고 문제 개수까지 하나 늘어난다.
    """
    return "skipped" in result


def _skipped_line(result: dict) -> _Line:
    """생략 사유 한 줄. 판정 줄이 아니므로 기호도 없고 어디에도 세지 않는다."""
    reason = result.get("skipped") or "생략됨"
    return _Line(
        "—",
        "dim",
        f"{reason}로 생략",
        is_problem=False,
        counts_as_item=False,
    )


def _status_line(result: dict) -> _Line:
    status = result.get("status")
    device = result.get("device")
    if status == "ok":
        return _Line("✔", "green", f"Canary 연산 실행    device={device} · 메모리 이동 확인됨")

    reason_key = f"status_{status}"
    message = _REASON_MESSAGES.get(reason_key, f"status={status} 감지 → 정상 실행 불가")
    detail = result.get("error_log")
    if detail:
        detail = _truncate_error_log(str(detail))
    return _Line("✖", "red", f"Canary 연산 실행    {message}", detail=detail, is_problem=True)


def _truncate_error_log(text: str, max_chars: int = _ERROR_LOG_MAX_CHARS) -> str:
    """긴 error_log를 "첫 프레임 + … + 꼬리"로 줄인다 — 꼬리(실제 예외 줄)는 항상 남긴다.

    예전에는 앞에서 200자만 남기고 잘랐는데, 파이썬 트레이스백은 진짜 원인
    (`ModuleNotFoundError: No module named 'torch'` 같은 예외 타입·메시지)이
    **마지막 줄**에 오므로 항상 그 줄이 잘려나가고 파일 경로만 남았다. 안내
    문구는 "에러 로그 확인 필요"라고 하는데 정작 그 로그가 잘려 있던 셈이다(#43).

    단순 tail로만 가면 "무엇을 하다가 죽었는지"(첫 프레임)를 잃으므로 둘 다
    남긴다: 첫 프레임 한 줄 → "…" → 예산 안에서 뒤쪽 줄들. 트레이스백 헤더
    ("Traceback (most recent call last):")는 정보가 없어 첫 프레임으로 건너뛴다.

    개행 없는 한 줄 로그(engine이 만드는 f"{type(exc).__name__}: {exc}" 등)는 예외
    이름이 맨 앞에 오므로 앞뒤를 반씩 남긴다 — 첫 프레임/마지막 줄 논리를 그대로
    태우면 같은 줄이 head와 tail에 중복돼 출력이 입력보다 길어진다(PR #48 리뷰).

    줄였을 때는 마지막에 안내 한 줄을 덧붙인다. 가장 짧은 실패 로그(torch 미설치)도
    230자라 사실상 모든 에러가 잘리는데, 잘렸다는 말이 없으면 사용자는 그게
    전부라고 믿는다. 전문은 --json에 그대로 있다.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text

    lines = text.splitlines()
    if len(lines) == 1:
        half = (max_chars - 1) // 2
        return f"{text[:half]}…{text[-half:]}\n{_TRUNCATION_NOTE}"

    head = lines[0]
    if head.startswith("Traceback (most recent call last)") and len(lines) > 1:
        head = lines[1]
    head = head.strip()
    head_budget = 80
    if len(head) > head_budget:
        head = head[: head_budget - 1] + "…"

    tail_budget = max(max_chars - len(head) - 1, 40)
    last_line = lines[-1].strip()
    if len(last_line) >= tail_budget:
        # 마지막 줄 자체가 예산보다 길면 그 줄만이라도 통째로 — 절대 앞을 자르지 않는다.
        tail = last_line
    else:
        start = len(text) - tail_budget
        # 줄 중간에서 시작하면 그 조각은 버리고 다음 줄부터 — 마지막 줄은 항상 남는다.
        if text[start - 1] != "\n":
            newline = text.find("\n", start)
            if newline != -1:
                start = newline + 1
        tail = text[start:]

    return f"{head}\n…\n{tail}\n{_TRUNCATION_NOTE}"


def _timing_line(result: dict) -> _Line:
    elapsed_ms = result.get("elapsed_ms")
    cpu_multiplier = result["cpu_multiplier"]
    reasons = result.get("reasons", [])
    if "cpu_multiplier_low" in reasons:
        return _Line(
            "⚠",
            "yellow",
            f"실행 시간 {elapsed_ms:.0f}ms    CPU 대비 {cpu_multiplier:.1f}배"
            " (2배 미만 — 성능 저하 가능성)",
            is_problem=True,
        )
    return _Line(
        "✔",
        "green",
        f"실행 시간 {elapsed_ms:.0f}ms    CPU 대비 {cpu_multiplier:.0f}배 (정상 범위)",
    )


def _format_memory_delta(memory_delta_mb: float) -> str:
    """canary 실측값을 읽기 좋은 단위로 — 1GB 미만은 MB, 이상은 GB.

    작은 모델(tiny-random-gpt2 등)은 18MB 정도만 옮기는데 `:.1f`GB로 찍으면
    `0.0GB`가 되어 **측정에 실패한 것처럼** 읽힌다. "실제로 재봤다"가 핵심
    가치인 도구가 정확히 그 지점에서 0.0을 보여주면 신뢰를 잃는다(#45).
    """
    if memory_delta_mb < 1:
        return "<1MB"
    if memory_delta_mb < 1024:
        return f"{memory_delta_mb:.0f}MB"
    return f"{memory_delta_mb / 1024:.1f}GB"


def _vram_line(result: dict) -> _Line:
    """canary가 실제로 옮긴 메모리량과, judge_result가 WARN 판정에 쓴 것과
    "같은" 가용/총 VRAM 숫자를 함께 보여준다.

    이전에는 최상위 `total_vram_gb`(GB)를 찾았는데 아무도 채워주지 않아 항상
    None이었고, 설상가상 채워졌더라도 judge_result가 WARN 판정에 쓰는 숫자는
    `env.gpu_free_mb`(MB, free 기준)라서 화면(total 기준)과 판정(free 기준)이
    서로 다른 숫자를 썼다 — WARN이 떠도 "12GB 중 8.4GB인데 왜 경고?"처럼
    화면만 봐서는 이유를 알 수 없는 문제가 있었다(PR #26 리뷰, 상영님 지적).
    이제 cli.py가 병합해줄 `env.gpu_free_mb`/`env.gpu_total_mb`(둘 다 MB)를
    그대로 읽어, 판정에 쓴 숫자와 화면 숫자를 일치시킨다.
    """
    memory_delta_mb = result.get("memory_delta_mb")
    if memory_delta_mb is None:
        return _Line("✔", "green", "VRAM 실측    측정값 없음")

    measured = _format_memory_delta(memory_delta_mb)
    env = result.get("env") or {}
    free_mb = env.get("gpu_free_mb")
    total_mb = env.get("gpu_total_mb")

    if free_mb is not None and total_mb is not None:
        # 가용·총은 항상 GB로 충분히 크므로 단위를 바꾸지 않는다(#45 범위 제외).
        detail = f"{measured} / {free_mb / 1024:.1f}GB 가용 (총 {total_mb / 1024:.0f}GB)"
    else:
        detail = f"{measured} 사용"

    return _Line("✔", "green", f"VRAM 실측    {detail}")


def _memory_headroom_line(result: dict) -> _Line | None:
    """judge.py의 "memory_delta_high"(가용 VRAM 90% 이상 소모) WARN을 화면에 반영한다.

    이 reason이 있는데도 보여줄 줄이 없으면 --model 모드에서는 `_vram_line`이
    항상 초록 ✔이라 WARN이 떠도 화면·문제 카운트 어디에도 안 보였고, 기본
    체크 모드는 애초에 VRAM 줄 자체가 없어 조용히 사라졌다(PR #26 리뷰 중
    자체 발견). reason이 없으면 None을 돌려줘 아무 줄도 추가하지 않는다.
    """
    if "memory_delta_high" not in result.get("reasons", []):
        return None

    memory_delta_mb = result.get("memory_delta_mb")
    free_mb = (result.get("env") or {}).get("gpu_free_mb")
    if memory_delta_mb is not None and free_mb:
        ratio_pct = memory_delta_mb / free_mb * 100
        detail = f"가용 VRAM의 {ratio_pct:.0f}% 소모 — 실제 학습 시 OOM 위험 높음"
    else:
        detail = _REASON_MESSAGES["memory_delta_high"]
    return _Line("⚠", "yellow", f"VRAM 여유    {detail}", is_problem=True)


def _quant_lines(result: dict) -> list[_Line]:
    """기본 체크 전용 줄(들)이다 — cli.md의 --model 예시에는 이 줄이 없다.

    --model 모드는 raw 스키마에 quant_backend가 있어도 4bit 레이어 자체를
    별도로 보여주지 않고 대신 VRAM·목표 크기 줄을 보여준다(_build_lines 참고).

    device=cpu FAIL과 quant_fallback 정보성 표시는 judge.py에서 독립 조건이라
    (#18) 동시에 참일 수 있다 — quant_backend="nn-linear-fallback" 인데
    device도 cpu인 환경(GPU도 4bit도 안 됨)은 이 두 줄이 함께 나와야
    "문제 있음"이 화면에서 사라지지 않는다. 리스트를 돌려주는 이유가 이거다.
    """
    reasons = result.get("reasons", [])
    quant_backend = result.get("quant_backend")
    device = result.get("device")
    lines: list[_Line] = []

    if "quant_layer_device_cpu" in reasons:
        lines.append(
            _Line(
                "✖",
                "red",
                "bitsandbytes 4bit 레이어    device=cpu 감지 → 조용한 CPU 폴백",
                is_problem=True,
            )
        )

    if quant_backend == "nn-linear-fallback" or "quant_fallback" in reasons:
        lines.append(
            _Line(
                "ℹ",
                "dim",
                f"4bit 레이어 폴백    {_REASON_MESSAGES['quant_fallback']}",
                is_problem=False,
            )
        )

    if not lines:
        # bnb-4bit + cuda: 문제도 폴백도 없는 정상 케이스만 여기 도달한다.
        lines.append(_Line("✔", "green", f"bitsandbytes 4bit 레이어    device={device} 정상"))

    return lines


def _target_size_line(result: dict) -> _Line | None:
    """--model 모드의 "목표 배치 크기 적합" 줄.

    raw 스키마(docs/contracts/canary-api.md)에는 batch_size/seq_len을 결과에
    되돌려주는 필드가 없다 — cli.py가 아직 이 값들을 result dict에 병합해주지
    않는 한(W9 이후 배선) 이 줄은 만들 수 없으므로, 없으면 조용히 생략한다.
    """
    batch_size = result.get("batch_size")
    seq_len = result.get("seq_len")
    if batch_size is None or seq_len is None:
        return None
    return _Line("✔", "green", f"목표 배치 크기 적합    batch={batch_size}, seq={seq_len} 기준")


def _is_model_mode(result: dict) -> bool:
    """--model 체크 결과인지를 `model_name` 존재 여부로만 판단한다.

    예전에는 `cpu_multiplier is None and status == "ok"`로 추론했는데,
    `cpu_multiplier`가 None이 되는 경로가 두 가지였다 — ① 실제 --model
    모드, ② GPU가 없는 "기본 체크"(worker.py가 device != "cuda"면 CPU
    배속 비교 자체를 생략). ②를 모델 모드로 오인하면 "4bit 레이어
    device=cpu" FAIL 줄이 통째로 안 그려지는 else 분기로 새 버려, 판정은
    FAIL(종료 코드 1)인데 화면은 "문제 없음"이라고 말하는 상황이 났다
    (#18에서 고친 문제가 리포트 층에서 재발 — PR #26 리뷰, 상영님 지적).
    `model_name`은 cli.py가 --model 체크 결과에만 병합해주는 값이라 모드를
    확정적으로 가른다 — 추론이 아니라 사실이다.
    """
    return result.get("model_name") is not None


def _build_lines(result: dict) -> list[_Line]:
    """result 하나로부터 화면에 찍힐 line item 목록을 만든다."""
    # 생략된 체크는 판정 필드가 통째로 없으므로 _status_line보다 먼저 걸러야 한다.
    if _is_skipped(result):
        return [_skipped_line(result)]

    lines: list[_Line] = [_status_line(result)]

    if result.get("status") != "ok":
        return lines

    if _is_model_mode(result):
        # --model 모드: VRAM 실측 + (가능하면) 목표 크기 적합 — quant 줄은 없다
        # (cli.md의 --model 예시 참고, 기본 체크 전용 줄이다).
        lines.append(_vram_line(result))
        target_line = _target_size_line(result)
        if target_line is not None:
            lines.append(target_line)
    else:
        # 기본 체크: quant 줄(_quant_lines)은 항상 그린다 — GPU가 아예 없는
        # 환경은 cpu_multiplier도 None이라(worker.py가 device != "cuda"면
        # CPU 배속 비교를 생략) `cpu_multiplier is not None`을 조건으로 걸면
        # "4bit 레이어 device=cpu" FAIL 줄까지 함께 사라진다 — 판정은 FAIL인데
        # 화면은 "문제 없음"이 되는 #18 재발 버그였다(PR #26 리뷰, 상영님 지적).
        # timing 줄만 실제로 잰 값이 있을 때로 조건을 좁힌다.
        if result.get("cpu_multiplier") is not None:
            lines.append(_timing_line(result))
        lines.extend(_quant_lines(result))

    headroom_line = _memory_headroom_line(result)
    if headroom_line is not None:
        lines.append(headroom_line)

    return lines


def _render_fix_block(console: Console, result: dict) -> None:
    fix = result.get("fix")
    if not fix:
        return
    fix_command = fix.get("fix_command")
    if fix_command:
        console.print(f"FIX: {fix_command}")
        console.print("재확인: preflight check --yes")
    else:
        message = fix.get("message", "")
        console.print(f"안내: {message}")


def _group_label(result: dict) -> str:
    """결과가 여러 개일 때(기본 체크 + --model 체크) 각 블록 앞에 붙일 표제.

    `_is_model_mode()`와 같은 기준(model_name 존재)으로 판단한다 — "index 0은
    항상 기본 체크"라는 순서 가정에 기대지 않는다(PR #26 리뷰, 상영님 지적:
    순서 추론은 결과가 하나뿐인 경우에는 아예 정보가 없고, 여럿이라도 실행
    순서가 바뀌면 깨진다).
    """
    label = f"모델 체크: {result['model_name']}" if _is_model_mode(result) else "기본 체크"
    if result.get("reverified"):
        # --yes로 수정을 실행한 뒤 다시 측정한 블록이다. 표시가 없으면 1차 실행
        # 결과와 구별되지 않아, 사용자는 화면의 ✔이 수정 덕분인지 원래 그랬는지
        # 알 수 없다(#57).
        label += " (재확인)"
    return label


def _render_text(results: list[dict], elapsed_seconds: float | None, notices: list[str]) -> None:
    console = Console()
    # 결과가 2개 이상(기본 체크 + --model 체크)일 때만 구분 표제를 붙인다 —
    # 단일 결과(기존 출력)는 지금까지의 화면 그대로 유지해 하위 호환을 지킨다.
    # 재확인한 블록만은 결과가 하나여도 표제를 붙인다 — "(재확인)"이라는 사실
    # 자체가 표제에만 있어서, 안 붙이면 수정 전후 화면이 똑같아진다.
    show_group_labels = len(results) > 1 or any(r.get("reverified") for r in results)

    all_lines: list[_Line] = []
    for index, result in enumerate(results):
        if show_group_labels:
            if index > 0:
                console.print()
            console.print(f"[bold]{_group_label(result)}[/bold]")

        lines = _build_lines(result)
        all_lines.extend(lines)
        for line in lines:
            # 판정 줄이 아닌 것(정보성 ℹ, 생략 —)은 줄 전체를 흐리게 — 판정 결과와
            # 시각적으로 섞이지 않게 한다.
            if line.symbol == "ℹ" or not line.counts_as_item:
                console.print(f"[dim]{line.symbol} {line.text}[/dim]")
            else:
                console.print(f"[{line.style}]{line.symbol}[/{line.style}] {line.text}")
            if line.detail:
                # error_log는 자유 텍스트라 "[stderr]" 같은 대괄호가 rich 마크업으로
                # 먹혀 사라지거나(실측), "[/x]" 꼴이면 MarkupError로 죽는다 — escape한다.
                console.print(f"    [dim]{escape(line.detail)}[/dim]")

    # FIX 블록은 개별 체크 블록 사이가 아니라 **전부 그린 뒤 한 번에** 나온다
    # (cli.md "출력 예시"). 체크 블록 사이에 끼면 "무엇이 문제인가"를 읽는 흐름이
    # "무엇을 해야 하는가"로 끊긴다 — 결과가 2개가 되면서 드러난 차이라, 결과가
    # 1개일 때의 화면은 이 변경 전후가 동일하다.
    for result in results:
        if result.get("verdict") != "PASS" and result.get("fix"):
            console.print()
            _render_fix_block(console, result)

    # --yes가 무엇을 했는지(또는 왜 아무것도 안 했는지)를 알리는 줄이다. FIX 블록
    # 다음, 요약 줄 앞에 온다 — "무엇이 문제인가 → 무엇을 해야 하는가 → 도구가
    # 실제로 무엇을 했는가" 순서다.
    if notices:
        console.print()
        for notice in notices:
            console.print(escape(notice))

    console.print()

    # 생략 줄(counts_as_item=False)은 판정된 적이 없어 항목 수에서 뺀다.
    item_count = sum(1 for line in all_lines if line.counts_as_item)
    problem_count = sum(1 for line in all_lines if line.is_problem)
    is_model_mode = len(results) == 1 and _is_model_mode(results[0])
    item_label = "1개 모델 확인" if is_model_mode else f"{item_count}개 항목 확인"
    problem_label = "문제 없음" if problem_count == 0 else f"{problem_count}개 문제 발견"

    summary = f"{item_label} · {problem_label}"
    if elapsed_seconds is not None:
        summary += f" · 소요 시간 {elapsed_seconds:.0f}초"
    console.print(summary)


def _compute_summary(results: list[dict], elapsed_seconds: float | None) -> dict:
    total_items = 0
    pass_count = 0
    warn_count = 0
    fail_count = 0
    for result in results:
        for line in _build_lines(result):
            if not line.counts_as_item:
                # 생략 줄 — 판정된 적이 없으므로 total에도 pass/warn/fail에도 안 센다.
                continue
            total_items += 1
            if not line.is_problem:
                if line.symbol == "✔":
                    pass_count += 1
                # info-only(ℹ) 라인은 pass/warn/fail 어디에도 세지 않는다.
            elif line.style == "yellow":
                warn_count += 1
            elif line.style == "red":
                fail_count += 1

    return {
        "total_items": total_items,
        "pass": pass_count,
        "warn": warn_count,
        "fail": fail_count,
        "elapsed_seconds": elapsed_seconds,
    }


def _render_json(results: list[dict], elapsed_seconds: float | None, notices: list[str]) -> None:
    summary = _compute_summary(results, elapsed_seconds)
    # 생략된 항목은 verdict 자체가 없다 — 판정에서 빼고 본다. 실제로는 생략이
    # 기본 체크 FAIL일 때만 일어나 결과가 뒤집히진 않지만, "우연히 맞는" 상태를
    # 남기지 않으려고 명시적으로 거른다(cli.md "결과 집계").
    judged = [r for r in results if not _is_skipped(r)]
    exit_code_hint = 0 if all(r.get("verdict") == "PASS" for r in judged) else 1
    payload = {
        "results": results,
        "summary": summary,
        "notices": notices,
        "exit_code_hint": exit_code_hint,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def render_report(
    results: list[dict],
    json_output: bool = False,
    elapsed_seconds: float | None = None,
    notices: list[str] | None = None,
) -> None:
    """results(judge_result() 출력 목록, 선택적으로 "fix" 키 병합)를 화면 또는 JSON으로 출력한다.

    `notices`는 판정이 아니라 **도구가 한 일**을 알리는 줄이다(`--yes`가 실행한
    명령, 실행할 명령이 없었다는 사실, 수정 실패). 텍스트 모드에서는 요약 줄 앞에,
    JSON 모드에서는 `notices` 배열로 나간다 — 자동화 쪽도 `--yes`가 실제로 뭘
    했는지 알아야 하기 때문이다(#53·#57).

    exit code 계산은 이 함수의 책임이 아니다 — 호출한 쪽(cli.py)이 results를 보고 직접
    판단한다(WARN/FAIL 구분은 cli.md에서도 아직 미확정). JSON 모드의 exit_code_hint는
    참고용 힌트일 뿐, 이 함수는 sys.exit을 호출하지 않는다.
    """
    if json_output:
        _render_json(results, elapsed_seconds, notices or [])
    else:
        _render_text(results, elapsed_seconds, notices or [])
