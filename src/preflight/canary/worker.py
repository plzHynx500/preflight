"""subprocess로 격리 실행되는 canary 본체 (forward+backward+optimizer.step()+측정).

engine.run_canary_check()이 `python -m preflight.canary.worker <spec> <result>`로 이
모듈을 별도 프로세스에서 기동한다. import 크래시·OOM이 나도 부모 프로세스는 죽지
않아야 한다 (docs/adr/0002-subprocess-isolation-for-canary.md 참고).

**이 파일이 아직 담당하지 않는 것** — 실패를 `import_crash`/`oom`으로 세분화하는
일은 W3(#2)의 범위다. 지금은 실행 중 어떤 예외가 나든 `status="error"`와 원본
traceback으로 정규화한다. W3에서 import 구간을 별도 try로 분리하고
`torch.cuda.OutOfMemoryError`를 예외 타입으로 잡아 세분화한다.
"""

from __future__ import annotations

import json
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
STATUS_ERROR = "error"

# 기본 체크 크기 (docs/architecture.md §3). 호출 측이 값을 주지 않을 때만 쓴다.
DEFAULT_BATCH_SIZE = 1
DEFAULT_SEQ_LEN = 8


def main() -> None:
    spec_path, result_path = sys.argv[1], sys.argv[2]
    with open(spec_path, encoding="utf-8") as spec_file:
        spec = json.load(spec_file)

    try:
        result = _run(spec)
    except Exception:  # noqa: BLE001 - 어떤 실패든 진단 결과로 포장해야 한다
        result = {
            "status": STATUS_ERROR,
            "device": None,
            "memory_delta_mb": None,
            "elapsed_ms": None,
            "cpu_multiplier": None,
            "quant_backend": None,
            "error_log": traceback.format_exc(),
        }

    with open(result_path, "w", encoding="utf-8") as result_file:
        json.dump(result, result_file, ensure_ascii=False)


def _run(spec: dict) -> dict:
    import torch

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
    except Exception:
        if not prefer_4bit:
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
