"""MODULE-01 CanaryEngine. docs/contracts/canary-api.md의 run_canary_check 계약을 구현한다.

이 모듈은 **torch를 import하지 않는다.** `import torch` 자체가 죽는 상황을 잡는 것이
프로세스 격리의 목적이므로(docs/adr/0002-subprocess-isolation-for-canary.md), 부모가
torch를 건드리는 순간 격리가 무력화된다. 모델 구성과 실행은 전부 자식 프로세스
(`canary/worker.py`)에서 일어나고, 이 모듈은 자식을 띄우고 결과를 정규화만 한다.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

#: `run_canary_check()`가 항상 돌려주는 필드. docs/contracts/canary-api.md와 일치해야 한다.
RESULT_FIELDS = (
    "status",
    "device",
    "memory_delta_mb",
    "elapsed_ms",
    "cpu_multiplier",
    "quant_backend",
    "error_log",
    "env",
    "rss_peak_mb",
)

VALID_STATUSES = ("ok", "oom", "import_crash", "error")

#: 자식이 응답하지 않을 때 무한정 매달리지 않도록 둔 상한. torch/bitsandbytes 최초
#: import와 CUDA 컨텍스트 빌드만으로 3~10초가 걸리고(docs/architecture.md §6-03),
#: `--model`은 config 조회까지 더해지므로 넉넉하게 잡았다.
WORKER_TIMEOUT_SEC = 600

#: 부모가 강제 종료되면(작업 관리자, `taskkill /F`, CI job kill 등) `_run_worker()`의
#: `finally`가 전혀 돌지 않아 `preflight-canary-*` workdir가 영구히 남는다(#132). 매
#: 실행 시작 시 이만큼(1시간) 지난 잔여물만 지워, 동시에 돌고 있는 다른 인스턴스의
#: workdir는 건드리지 않는다. ADR-0002의 "부모가 죽어도 자식은 살아남아야 한다"는
#: 결정을 뒤집지 않는 가장 얕은 보완책이다 — Job Object로 자식까지 묶는 방안은 그
#: 결정과 상충할 수 있어 별도 설계 판단이 필요하므로 이번 범위에 넣지 않았다.
STALE_WORKDIR_AGE_SEC = 3600


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
    _cleanup_stale_workdirs()
    workdir = tempfile.mkdtemp(prefix="preflight-canary-")
    spec_path = os.path.join(workdir, "spec.json")
    result_path = os.path.join(workdir, "result.json")
    try:
        spec = {"model_name": model_name, "batch_size": batch_size, "seq_len": seq_len}
        with open(spec_path, "w", encoding="utf-8") as spec_file:
            json.dump(spec, spec_file, ensure_ascii=False)

        try:
            completed = subprocess.run(
                [sys.executable, "-m", "preflight.canary.worker", spec_path, result_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=WORKER_TIMEOUT_SEC,
                check=False,
            )
        except subprocess.TimeoutExpired:
            # subprocess.run이 자식을 종료시킨 뒤 올라온다. 자식이 어느 단계까지
            # 갔었는지는 그때까지 기록해둔 결과 파일이 알려준다.
            return _normalize(
                _read_result(result_path)
                or {
                    "status": "error",
                    "error_log": (
                        f"canary 자식 프로세스가 {WORKER_TIMEOUT_SEC}초 안에 끝나지 않아 "
                        "강제 종료했다."
                    ),
                }
            )

        raw = _read_result(result_path)
        if raw is None:
            # 자식이 사전 기록조차 남기지 못한 경우 — 프로세스 기동 자체가 실패했거나
            # 결과 파일을 쓸 수 없는 상황이다.
            return _normalize({"status": "error", "error_log": _crash_log(completed)})
        return _normalize(raw)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _cleanup_stale_workdirs() -> None:
    """이전 실행이 강제 종료돼 남긴 잔여 workdir를 지운다(#132).

    청소 자체가 실패해도 지금 실행하려는 canary 체크를 막으면 안 되므로, 무엇이 나든
    삼킨다 — 여기서 새는 예외가 `run_canary_check()`의 바깥 `try`에 잡히면 정상적인
    체크 실패(`status="error"`)로 오진된다.
    """
    try:
        pattern = os.path.join(tempfile.gettempdir(), "preflight-canary-*")
        now = time.time()
        for stale_dir in glob.glob(pattern):
            try:
                if now - os.path.getmtime(stale_dir) >= STALE_WORKDIR_AGE_SEC:
                    shutil.rmtree(stale_dir, ignore_errors=True)
            except OSError:
                continue
    except Exception:  # noqa: BLE001, S110 - 정리 실패가 canary 체크 자체를 막으면 안 된다
        pass


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
    # `env`는 소비자(causes.py 등)가 곧바로 .get()으로 파고드는 자리라, dict가 아닌
    # 값이 흘러들면 부모가 그 자리에서 죽는다. 타입이 어긋나면 없는 것으로 본다.
    if not isinstance(result["env"], dict):
        result["env"] = None
    if result["status"] not in VALID_STATUSES:
        result["error_log"] = "예상치 못한 status: {!r}\n{}".format(
            result["status"], result["error_log"] or ""
        ).strip()
        result["status"] = "error"
    return result
