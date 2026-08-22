"""MODULE-02 FixExecutor. docs/contracts/canary-api.md의 suggest_fix 계약을 구현한다."""

from __future__ import annotations

import shlex
import subprocess
import sys

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


#: cause -> (사람이 읽는 원인 문구, `sys.executable` 뒤에 붙일 인자들 또는 None).
#:
#: 명령을 문자열이 아니라 **인자 리스트**로 들고 있는 이유가 두 가지다(#52).
#: ① 실행 파이썬을 실행 시점에 끼워 넣어야 한다 — 아래 `_build_command` 참고.
#: ② Windows의 `C:\Program Files\...\python.exe`처럼 공백·역슬래시가 든 경로를
#:    문자열로 조립했다가 `shlex.split`으로 되돌리면 인용이 깨진다. 조립하지 않으면
#:    되돌릴 일도 없다.
_FIX_MAP: dict[str, tuple[str, list[str] | None]] = {
    "bnb_not_compiled_with_cuda": (
        "bitsandbytes가 CUDA 지원 없이 빌드됨",
        ["-m", "pip", "install", "bitsandbytes", "--upgrade", "--force-reinstall"],
    ),
    "import_crash_general": (
        "PyTorch/CUDA 또는 라이브러리 import 실패 (에러 로그 확인 필요)",
        None,
    ),
    "oom": (
        (
            "CUDA Out of Memory: 목표 모델/배치 크기 실행 불가"
            " (batch_size 축소 또는 quantization 적용 필요)"
        ),
        None,
    ),
    "4bit_cpu_fallback_other": (
        "4bit 양자화 레이어가 CPU로 분기됨 (CUDA 메모리 부족 또는 지원 불가 장치)",
        None,
    ),
    "memory_delta_high": (
        (
            "canary 실행만으로 가용 VRAM의 90% 이상 소모 — 실제 학습 시 배치/모델 크기를"
            " 그대로 쓰면 OOM 가능성 높음 (batch_size 축소 권장)"
        ),
        None,
    ),
    "cpu_multiplier_low": (
        "CPU 대비 연산 속도 2배 미만 (원인 특정 어려운 회색지대 — 성능 저하 가능성 안내)",
        None,
    ),
    "unknown_error": ("Canary 실행 오류 (설정 또는 환경 확인 필요)", None),
    "unknown": ("알 수 없는 이상 진단 결과", None),
}


def _quote(arg: str) -> str:
    """화면에 찍을 때만 쓰는 최소 인용 — 공백이 있으면 큰따옴표로 감싼다.

    `shlex.quote`는 POSIX 규칙이라 Windows 경로를 작은따옴표로 감싸는데,
    cmd.exe/PowerShell은 그걸 인용으로 보지 않는다. 큰따옴표는 세 셸에서 모두
    통해서 "그대로 복사해 붙일 수 있다"는 조건을 지킨다.
    """
    return f'"{arg}"' if not arg or any(ch.isspace() for ch in arg) else arg


def _build_command(args: list[str] | None) -> tuple[list[str] | None, str | None]:
    """`sys.executable`을 앞에 붙여 실행용 argv와 표시용 문자열을 함께 만든다.

    PATH에서 찾은 `pip`을 쓰면 **진단한 환경과 다른 환경에 설치될 수 있다**(#52).
    venv를 활성화하지 않고 절대경로로 실행했거나, pipx/전역 설치로 preflight를
    쓰거나, conda 환경과 PATH가 어긋난 경우가 전부 여기 해당한다 — 그리고 그런
    사람이 바로 이 도구의 표적이다. 진단은 A 환경을 보고 수정은 B 환경에 하면
    재확인이 계속 실패하는데 사용자는 이유를 알 수 없다.

    표시용 문자열에도 `sys.executable`의 전체 경로를 그대로 쓴다. 짧게 보이려고
    `python -m pip`으로 줄이면, 화면에서 복사해 붙이는 순간 다시 "지금 활성화된
    파이썬"으로 돌아가 같은 버그를 재현하게 된다.
    """
    if not args:
        return None, None
    argv = [sys.executable, *args]
    return argv, " ".join(_quote(a) for a in argv)


def suggest_fix(check_result: dict) -> dict | None:
    """실행하지 않는다 — 명령어 텍스트만 반환. verdict가 PASS면 None.

    **여기서 bitsandbytes를 import하지 않는다** (Issue #24). 이 함수는 부모
    프로세스에서 도는데, 원인 확인이 가장 필요한 상황이 곧 그 import가 가장
    위험한 상황이다 — `.so` 로드 실패는 파이썬 예외가 아니라 SIGSEGV로 나므로
    `try/except`로 막을 수도 없다. 자식이 `env`에 실어 보낸 값을 읽는다
    (ADR-0002, canary-api.md의 `env` 절).
    """
    if check_result.get("verdict") == "PASS":
        return None

    cause = classify_cause(check_result)
    if cause == "pass":
        return None

    message, args = _FIX_MAP.get(cause, _FIX_MAP["unknown"])
    fix_argv, fix_command = _build_command(args)
    return {
        "cause": cause,
        "message": message,
        "fix_command": fix_command,
        "fix_argv": fix_argv,
    }


def apply_fix(fix: dict) -> None:
    """--yes 지정 시에만 호출된다.

    실행에는 `fix_argv`(인자 리스트)를 쓴다 — 화면에 찍히는 `fix_command`
    문자열을 다시 파싱하지 않는다(#52). `fix_argv`가 없으면 예전 계약대로
    `fix_command`를 `shlex.split`해서 쓴다.

    표준 출력/에러를 캡처하고 실패 시 FixExecutionError를 발생시킨다.
    """
    argv = fix.get("fix_argv")
    cmd = fix.get("fix_command")
    if not argv:
        if not cmd:
            return
        argv = shlex.split(cmd)
    cmd = cmd or " ".join(_quote(a) for a in argv)

    try:
        subprocess.run(
            argv,
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
