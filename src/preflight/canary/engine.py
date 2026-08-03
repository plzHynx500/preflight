"""MODULE-01 CanaryEngine. docs/contracts/canary-api.md의 run_canary_check 계약을 구현한다.

이 모듈은 **torch를 import하지 않는다.** `import torch` 자체가 죽는 상황을 잡는 것이
프로세스 격리의 목적이므로(docs/adr/0002-subprocess-isolation-for-canary.md), 부모가
torch를 건드리는 순간 격리가 무력화된다. 모델 구성과 실행은 전부 자식 프로세스
(`canary/worker.py`)에서 일어나고, 이 모듈은 자식을 띄우고 결과를 정규화만 한다.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

#: `run_canary_check()`가 항상 돌려주는 필드. docs/contracts/canary-api.md와 일치해야 한다.
RESULT_FIELDS = (
    "status",
    "device",
    "memory_delta_mb",
    "elapsed_ms",
    "cpu_multiplier",
    "quant_backend",
    "error_log",
)

VALID_STATUSES = ("ok", "oom", "import_crash", "error")

#: 자식이 응답하지 않을 때 무한정 매달리지 않도록 둔 상한. torch/bitsandbytes 최초
#: import와 CUDA 컨텍스트 빌드만으로 3~10초가 걸리고(docs/architecture.md §6-03),
#: `--model`은 config 조회까지 더해지므로 넉넉하게 잡았다.
WORKER_TIMEOUT_SEC = 600


def run_canary_check(model_name: str | None, batch_size: int, seq_len: int) -> dict:
    """canary/worker.py를 subprocess로 격리 실행하고 원시 측정값을 반환한다.

    반환 스키마(status/device/memory_delta_mb/elapsed_ms/cpu_multiplier/quant_backend/
    error_log)는 docs/contracts/canary-api.md 참고. 자식이 어떻게 죽든 예외를 던지지
    않고 정규화된 dict를 돌려준다.

    `model_name`이 None이면 기본 체크(모델과 무관한 최소 대표 구조)를 실행한다.
    모델명을 준 경우의 실제 모델 구성은 W4(FR-02)에서 채워진다.
    """
    try:
        return _run_worker(model_name, batch_size, seq_len)
    except Exception as exc:  # noqa: BLE001 - 어떤 실패든 진단 결과로 포장해야 한다
        return _normalize({"status": "error", "error_log": f"{type(exc).__name__}: {exc}"})


def _run_worker(model_name: str | None, batch_size: int, seq_len: int) -> dict:
    workdir = tempfile.mkdtemp(prefix="preflight-canary-")
    spec_path = os.path.join(workdir, "spec.json")
    result_path = os.path.join(workdir, "result.json")
    try:
        spec = {"model_name": model_name, "batch_size": batch_size, "seq_len": seq_len}
        with open(spec_path, "w", encoding="utf-8") as spec_file:
            json.dump(spec, spec_file, ensure_ascii=False)

        completed = subprocess.run(
            [sys.executable, "-m", "preflight.canary.worker", spec_path, result_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=WORKER_TIMEOUT_SEC,
            check=False,
        )

        raw = _read_result(result_path)
        if raw is None:
            # 자식이 결과를 남기지 못하고 죽은 경우(.so 로드 실패로 인한 즉사 등).
            # 어느 단계에서 죽었는지 구분하는 일은 W3(#2)에서 다룬다.
            return _normalize({"status": "error", "error_log": _crash_log(completed)})
        return _normalize(raw)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _read_result(result_path: str):
    try:
        with open(result_path, encoding="utf-8") as result_file:
            payload = json.load(result_file)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _crash_log(completed: subprocess.CompletedProcess) -> str:
    parts = [
        f"canary 자식 프로세스가 결과를 남기지 못하고 종료됨 (exit code {completed.returncode})"
    ]
    for label, stream in (("stdout", completed.stdout), ("stderr", completed.stderr)):
        text = (stream or "").strip()
        if text:
            parts.append(f"[{label}]\n{text}")
    return "\n\n".join(parts)


def _normalize(raw: dict) -> dict:
    """자식이 준 값을 계약 스키마에 정확히 맞춘다 — 필드가 넘치지도 모자라지도 않게."""
    result = {field: raw.get(field) for field in RESULT_FIELDS}
    if result["status"] not in VALID_STATUSES:
        result["error_log"] = "예상치 못한 status: {!r}\n{}".format(
            result["status"], result["error_log"] or ""
        ).strip()
        result["status"] = "error"
    return result
