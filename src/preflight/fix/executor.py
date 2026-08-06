"""MODULE-02 FixExecutor. docs/contracts/canary-api.md의 suggest_fix 계약을 구현한다."""

from __future__ import annotations

import shlex
import subprocess

from preflight.fix.causes import classify_cause


class FixExecutionError(RuntimeError):
    """수정 명령어 실행 실패 시 발생한다."""

    def __init__(
        self,
        command: str,
        returncode: int | None,
        stdout: str,
        stderr: str,
        message: str = "",
    ) -> None:
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        msg = message or (
            f"수정 명령어 실행 실패 ('{command}'): exit code {returncode}\n"
            f"[stderr] {stderr.strip()}\n"
            f"[stdout] {stdout.strip()}"
        )
        super().__init__(msg)


_FIX_MAP: dict[str, dict[str, str | None]] = {
    "bnb_not_compiled_with_cuda": {
        "message": "bitsandbytes가 CUDA 지원 없이 빌드됨",
        "fix_command": "pip install bitsandbytes --upgrade --force-reinstall",
    },
    "import_crash_general": {
        "message": "PyTorch/CUDA 또는 라이브러리 import 실패 (에러 로그 확인 필요)",
        "fix_command": None,
    },
    "oom": {
        "message": "CUDA Out of Memory: 목표 모델/배치 크기 실행 불가 (batch_size 축소 또는 quantization 적용 필요)",
        "fix_command": None,
    },
    "4bit_cpu_fallback_other": {
        "message": "4bit 양자화 레이어가 CPU로 분기됨 (CUDA 메모리 부족 또는 지원 불가 장치)",
        "fix_command": None,
    },
    "memory_delta_high": {
        "message": "메모리 사용량이 예측 대비 15% 이상 벗어남 (fragmentation·activation 재계산 등 후보 원인 안내)",
        "fix_command": None,
    },
    "cpu_multiplier_low": {
        "message": "CPU 대비 연산 속도 2배 미만 (원인 특정 어려운 회색지대 — 성능 저하 가능성 안내)",
        "fix_command": None,
    },
    "unknown_error": {
        "message": "Canary 실행 오류 (설정 또는 환경 확인 필요)",
        "fix_command": None,
    },
    "unknown": {
        "message": "알 수 없는 이상 진단 결과",
        "fix_command": None,
    },
}


def suggest_fix(check_result: dict) -> dict | None:
    """실행하지 않는다 — 명령어 텍스트만 반환. verdict가 PASS면 None."""
    if check_result.get("verdict") == "PASS":
        return None

    if "compiled_with_cuda" not in check_result:
        try:
            import bitsandbytes.cextension.lib as bnb_lib

            check_result["compiled_with_cuda"] = getattr(bnb_lib, "compiled_with_cuda", False)
        except (ImportError, AttributeError):
            check_result["compiled_with_cuda"] = False

    cause = classify_cause(check_result)
    if cause == "pass":
        return None

    info = _FIX_MAP.get(cause, _FIX_MAP["unknown"])
    return {
        "cause": cause,
        "message": str(info["message"]),
        "fix_command": info["fix_command"],
    }


def apply_fix(fix: dict) -> None:
    """--yes 지정 시에만 호출된다.

    fix 딕셔너리의 'fix_command'를 실행하며,
    표준 출력/에러를 캡처하고 실패 시 FixExecutionError를 발생시킨다.
    """
    cmd = fix.get("fix_command")
    if not cmd:
        return

    args = shlex.split(cmd)
    try:
        subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise FixExecutionError(
            command=cmd,
            returncode=e.returncode,
            stdout=e.stdout or "",
            stderr=e.stderr or "",
        ) from e
    except OSError as e:
        raise FixExecutionError(
            command=cmd,
            returncode=None,
            stdout="",
            stderr=str(e),
            message=f"수정 명령어 실행 실패 ('{cmd}'): {e}",
        ) from e
