"""subprocess로 격리 실행되는 canary 본체 (forward+backward+optimizer.step()+측정).

engine.run_canary_check()이 `python -m preflight.canary.worker <spec> <result>`로 이
모듈을 별도 프로세스에서 기동한다. import 크래시·OOM이 나도 부모 프로세스는 죽지
않아야 한다 (docs/adr/0002-subprocess-isolation-for-canary.md 참고).

## 죽는 방식이 두 가지라 대비도 두 겹이다

**① 파이썬 예외로 잡히는 실패** — 구간별 `try`로 잡는다. import 구간에서 터지면
`import_crash`, 실행 구간의 `torch.cuda.OutOfMemoryError`면 `oom`, 그 외는 `error`다.
어느 예외 타입이냐가 아니라 **어느 구간에서 터졌느냐**로 나누므로, 에러 메시지
문자열을 뒤지지 않는다 (docs/adr/0002 "텍스트 파싱이 아니라 안정된 속성으로").

**② 프로세스가 통째로 죽는 실패** — `.so` 로드 실패는 파이썬 예외를 만들 기회조차
없이 SIGSEGV로 즉사한다. `except`도 `finally`도 돌지 않으므로 "죽은 뒤에 기록"이
불가능하다. 그래서 **죽기 전에 미리 기록**한다 — 자식은 시작하자마자 "여기서 죽으면
이게 정답"인 결과를 써두고, 단계를 통과할 때마다 덮어쓴다. 어느 시점에 죽든 마지막
기록이 남으므로 부모가 exit code를 해석할 필요가 없다(OS마다 다르다).
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

from preflight.canary.model import (
    QUANT_BACKEND_4BIT,
    build_dummy_model,
    build_minimal_canary_input,
    build_minimal_canary_model,
)

STATUS_OK = "ok"
STATUS_OOM = "oom"
STATUS_IMPORT_CRASH = "import_crash"
STATUS_ERROR = "error"

# 기본 체크 크기 (docs/architecture.md §3). 호출 측이 값을 주지 않을 때만 쓴다.
DEFAULT_BATCH_SIZE = 1
DEFAULT_SEQ_LEN = 8

_PREWRITE_IMPORT_NOTE = (
    "canary 스택(torch/bitsandbytes)을 import하는 도중 프로세스가 예외 없이 종료됐다. "
    "CUDA 라이브러리 .so 로드 실패로 인한 즉사가 대표적인 경우다."
)
_PREWRITE_RUN_NOTE = (
    "canary 실행 도중 프로세스가 예외 없이 종료됐다. import는 통과한 상태였다. "
    "호스트 RAM 부족으로 인한 강제 종료, 드라이버 크래시 등이 후보다 — "
    "VRAM 부족(OOM)은 보통 예외로 잡히므로 이 경로로 오지 않는다."
)


def main() -> None:
    spec_path, result_path = sys.argv[1], sys.argv[2]

    # 죽기 전에 미리 기록한다. 지금부터 import를 통과하기 전까지 프로세스가 즉사하면
    # 이 결과가 그대로 부모에게 읽힌다.
    _write_result(result_path, _blank_result(STATUS_IMPORT_CRASH, _PREWRITE_IMPORT_NOTE))

    try:
        with open(spec_path, encoding="utf-8") as spec_file:
            spec = json.load(spec_file)
    except Exception:  # noqa: BLE001 - spec을 못 읽는 것도 진단 결과로 포장한다
        _write_result(result_path, _blank_result(STATUS_ERROR, traceback.format_exc()))
        return

    try:
        torch = _import_canary_stack()
    except BaseException:  # noqa: BLE001 - SystemExit까지 포함해 import 실패로 본다
        _write_result(result_path, _blank_result(STATUS_IMPORT_CRASH, traceback.format_exc()))
        return

    # import를 통과했다 — 여기서부터 즉사하면 더 이상 import 문제가 아니다.
    _write_result(result_path, _blank_result(STATUS_ERROR, _PREWRITE_RUN_NOTE))

    try:
        result = _run(torch, spec)
    except Exception as exc:  # noqa: BLE001 - 어떤 실패든 진단 결과로 포장해야 한다
        status = STATUS_OOM if _is_oom(torch, exc) else STATUS_ERROR
        result = _blank_result(status, traceback.format_exc())

    _write_result(result_path, result)


def _import_canary_stack():
    """canary 스택을 미리 import해 즉사 가능 구간을 앞쪽에 몰아둔다.

    bitsandbytes가 **파이썬 예외로** 실패하는 경우(구버전·빌드 문제)는 크래시가
    아니라 폴백 대상이므로 여기서 삼킨다 — model.py가 다시 시도하며 `nn.Linear`로
    대체하고 그 사실을 `quant_backend`로 알린다 (docs/architecture.md §6-01).
    반대로 .so 로드 실패로 프로세스가 즉사하면 main()이 미리 써둔 import_crash가
    그대로 남는다.
    """
    import torch

    _try_import_bitsandbytes()
    return torch


def _try_import_bitsandbytes() -> bool:
    try:
        import bitsandbytes  # noqa: F401
    except Exception:  # noqa: BLE001 - 실패 종류와 무관하게 폴백에 맡긴다
        return False
    return True


def _blank_result(status: str, error_log: str | None) -> dict:
    """측정값이 없는 실패 결과. 계약 스키마(docs/contracts/canary-api.md)를 채운다."""
    return {
        "status": status,
        "device": None,
        "memory_delta_mb": None,
        "elapsed_ms": None,
        "cpu_multiplier": None,
        "quant_backend": None,
        "error_log": error_log,
    }


def _write_result(result_path: str, payload: dict) -> None:
    """원자적으로 기록한다 — 임시 파일에 다 쓴 뒤 이름만 바꾼다.

    `open(path, "w")`는 여는 순간 기존 내용을 먼저 지운다. 지운 뒤 새 내용을 쓰기
    전에 프로세스가 죽으면 **직전에 미리 써둔 정보까지 함께 날아간다.** rename은
    파일시스템이 보장하는 원자적 연산이라 "반쯤 바뀐" 상태가 없으므로, 실패해도
    이전 기록이 그대로 살아남는다.
    """
    tmp_path = result_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as tmp_file:
        json.dump(payload, tmp_file, ensure_ascii=False)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
    os.replace(tmp_path, result_path)


def _is_oom(torch, exc: BaseException) -> bool:
    """VRAM 부족 예외인지 **타입으로** 판별한다 — 메시지 문자열은 보지 않는다.

    `torch.cuda.OutOfMemoryError`는 `RuntimeError`의 자식이라 일반 예외보다 먼저
    걸러야 한다. 이 타입이 없는 구버전 torch에서는 판별하지 않고 `error`로 둔다 —
    추측해서 `oom`이라고 적으면 사용자가 batch를 줄이라는 엉뚱한 안내를 받는다.
    """
    oom_type = getattr(torch, "OutOfMemoryError", None)
    if oom_type is None:
        oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
    return oom_type is not None and isinstance(exc, oom_type)


def _run(torch, spec: dict) -> dict:
    model_name = spec.get("model_name")
    batch_size = int(spec.get("batch_size") or DEFAULT_BATCH_SIZE)
    seq_len = int(spec.get("seq_len") or DEFAULT_SEQ_LEN)

    if model_name is None:
        return _run_basic_check(torch, batch_size, seq_len)

    # `--model` 경로(FR-02)는 이인수의 W4가 build_dummy_model()을 채운 뒤 W9에서
    # 연결된다. 지금은 아래 호출이 NotImplementedError를 올리고 main()이 그것을
    # status="error" + error_log로 정규화한다 — 부모는 죽지 않는다.
    build_dummy_model(model_name)
    raise NotImplementedError("`--model` 경로는 아직 엔진에 연결되지 않았다 (WORKPLAN W4·W9).")


def _run_basic_check(torch, batch_size: int, seq_len: int) -> dict:
    """기본 체크 — 모델과 무관하게 GPU/드라이버/CUDA 체인이 살아있는지 실측한다."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # CPU는 float16 연산 유닛이 사실상 없어 정상 경로인 float32로 돌린다
    # (docs/architecture.md §6-01 "dtype 비대칭" 참고).
    dtype = torch.float16 if device == "cuda" else torch.float32

    measured = _measure(torch, device, dtype, batch_size, seq_len)

    cpu_multiplier = None
    if device == "cuda" and measured["elapsed_ms"]:
        baseline_ms = _measure_cpu_baseline(torch, batch_size, seq_len, measured["quant_backend"])
        if baseline_ms is not None:
            cpu_multiplier = baseline_ms / measured["elapsed_ms"]

    return {
        "status": STATUS_OK,
        "device": measured["device"],
        "memory_delta_mb": measured["memory_delta_mb"],
        "elapsed_ms": measured["elapsed_ms"],
        "cpu_multiplier": cpu_multiplier,
        "quant_backend": measured["quant_backend"],
        "error_log": None,
    }


def _measure_cpu_baseline(torch, batch_size: int, seq_len: int, quant_backend: str):
    """CPU 강제 폴백 실행 시간(ms). 측정에 실패하면 None을 돌려준다.

    절대 시간은 GPU 세대마다 크게 달라 쓸 수 없으므로, 이 값 대비 몇 배 빠른가로
    판정한다 (docs/adr/0003-relative-baseline-timing.md).
    """
    try:
        baseline = _measure(
            torch,
            "cpu",
            torch.float32,
            batch_size,
            seq_len,
            prefer_4bit=quant_backend == QUANT_BACKEND_4BIT,
        )
    except Exception:  # noqa: BLE001 - 기준선 실패가 canary 실패로 번지면 안 된다
        # 기준선을 못 재는 것은 canary 자체의 실패가 아니다 — 배수만 포기한다.
        return None
    return baseline["elapsed_ms"]


def _measure(torch, device: str, dtype, batch_size: int, seq_len: int, prefer_4bit: bool = True):
    try:
        return _measure_once(torch, device, dtype, batch_size, seq_len, prefer_4bit)
    except Exception as exc:
        # OOM은 폴백으로 해결되지 않는다 — nn.Linear로 다시 돌려도 메모리는 그대로
        # 부족하다. 재시도하면 시간만 버리고 원래 원인(oom)도 흐려지므로 그대로 올린다.
        if not prefer_4bit or _is_oom(torch, exc):
            raise
        # 구버전 bitsandbytes 등으로 4bit 경로가 통째로 실패하면 평범한 nn.Linear로
        # 한 번만 더 시도한다 (docs/architecture.md §6-01 "구버전 bnb 대응").
        return _measure_once(torch, device, dtype, batch_size, seq_len, False)


def _measure_once(torch, device: str, dtype, batch_size: int, seq_len: int, prefer_4bit: bool):
    """canary를 2회 실행해 1회차(워밍업)를 버리고 2회차만 측정한다.

    측정 방식의 근거는 docs/architecture.md §4 "실행 횟수"·"측정 방법" 참고.
    """
    model, quant_backend = build_minimal_canary_model(device, dtype, prefer_4bit)
    dummy_input = build_minimal_canary_input(batch_size, seq_len, device, dtype)
    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)

    # 1회차 — CUDA 컨텍스트 생성·cuBLAS 초기화·allocator 첫 할당 비용을 여기서 태운다.
    _train_step(model, dummy_input, optimizer)

    if device == "cuda":
        torch.cuda.synchronize()
        # 최고점 기준을 여기서 리셋한다. max_memory_allocated()는 증가분이 아니라
        # 총 할당량의 최고점이라, 워밍업에서 이미 잡혀 살아 있는 옵티마이저 모멘텀
        # 버퍼와 가중치도 그대로 포함된다.
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _train_step(model, dummy_input, optimizer)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        memory_delta_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        started_at = time.perf_counter()
        _train_step(model, dummy_input, optimizer)
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        # CUDA가 없으면 잴 VRAM도 없다. device=cpu 자체가 판정 대상이다.
        memory_delta_mb = None

    return {
        "device": _base_layer_device(model),
        "elapsed_ms": elapsed_ms,
        "memory_delta_mb": memory_delta_mb,
        "quant_backend": quant_backend,
    }


def _train_step(model, dummy_input, optimizer) -> None:
    """forward → backward → optimizer.step().

    `step()`까지 반드시 돌린다 — Adam 계열은 이 호출 전까지 모멘텀 버퍼를 만들지
    않아, forward+backward까지만 하면 옵티마이저 메모리 풀을 통째로 놓친다
    (docs/architecture.md §4).
    """
    optimizer.zero_grad(set_to_none=True)
    loss = model(dummy_input).float().pow(2).mean()
    loss.backward()
    optimizer.step()


def _base_layer_device(model):
    """4bit 베이스 레이어 파라미터가 실제로 올라가 있는 device.

    "4bit 레이어 device=cpu 감지"가 판정 항목이므로 모델 전체가 아니라 베이스
    레이어를 직접 본다.
    """
    for param in model[0].base.parameters():
        return str(param.device).split(":")[0]
    return None


if __name__ == "__main__":
    main()
