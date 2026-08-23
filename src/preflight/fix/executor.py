"""MODULE-02 FixExecutor. docs/contracts/canary-api.md의 suggest_fix 계약을 구현한다."""

from __future__ import annotations

import re
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


#: CPU 전용 torch를 CUDA 빌드로 갈아끼우는 인자를 만든다 — `_build_command`에 넘겨
#: `sys.executable`을 앞세운다(#52).
#:
#: `--force-reinstall`이 필요하다 — 없으면 pip이 "torch는 이미 설치돼 있다"며 아무
#: 것도 하지 않고, `--yes`가 성공한 것처럼 끝난 뒤 재검증에서 같은 실패가 나온다.
def _torch_cuda_reinstall_args(tag: str) -> list[str]:
    return [
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "torch",
        "--index-url",
        f"https://download.pytorch.org/whl/{tag}",
    ]


#: 드라이버가 없거나 못 읽었을 때, 또는 아래 표의 어떤 구간에도 안 들 때 떨어지는
#: 기본 휠 태그다. cu124는 비교적 오래된 CUDA 12.4용이라 최신 드라이버에서도 대체로
#: 동작한다(#82 원안의 기존 고정값을 그대로 유지).
_DEFAULT_TORCH_CUDA_TAG = "cu124"
_TORCH_CUDA_REINSTALL_DISPLAY = (
    "pip install --force-reinstall torch --index-url"
    f" https://download.pytorch.org/whl/{_DEFAULT_TORCH_CUDA_TAG}"
)

#: 드라이버 **major 브랜치 번호**(버전 문자열 첫 자리, 예: "595.79" → 595) → 골라줄
#: PyTorch CUDA 휠 태그. 내림차순으로 두고 처음 만족하는 구간을 쓴다.
#:
#: patch 단위까지는 못 맞춘다. `env.gpu_driver_version`은 NVML이 OS 그대로 돌려준
#: 문자열이라 Linux는 세 자리("560.28.03"), Windows는 두 자리("560.76")로 형식이
#: 다르고, NVIDIA의 CUDA Toolkit 최소 드라이버 표도 OS마다 patch 값이 다르다
#: (2026-08 조사). 두 표 모두 같은 major 브랜치 번호(560/570/580 …)를 쓰길래
#: 그 자리만 비교한다 — 경계에 걸친 몇 안 되는 드라이버는 기본값으로 떨어질 뿐이라
#: 틀리더라도 안전한 쪽(더 낮은 CUDA 요구치)으로만 틀린다.
#:
#: cu121·cu124·cu128·cu129는 후보에서 뺐다 — download.pytorch.org 실측(2026-08)으로
#: 이 태그들이 특정 torch 버전에서 동결돼 있다(cu124→2.6.0 고정, cu129→2.9.0 고정 등).
#: "드라이버가 지원하는 가장 높은 CUDA"를 그대로 고르면 오히려 지금 받을 수 있는
#: 최신 torch(2.13.0)보다 낮은 버전을 설치하는 역효과가 난다. cu126·cu130만 계속
#: 최신 torch 빌드를 받는다.
#:
#: **이 표는 시간이 지나면 어긋난다.** PyTorch가 태그를 새로 열거나 동결할 때,
#: NVIDIA가 새 CUDA 메이저를 낼 때 다시 확인해야 한다 — 근거와 재확인 기준은
#: docs/adr/0007-driver-version-based-torch-cuda-wheel-selection.md 참고.
_TORCH_CUDA_TAG_BY_MIN_DRIVER_MAJOR: list[tuple[int, str]] = [
    (580, "cu130"),  # CUDA 13.0 GA, Linux 최소 580.65.06
    (560, "cu126"),  # CUDA 12.6 GA, Linux 560.28.03 / Windows 560.76
]


def _torch_cuda_tag_for_driver(driver_version: str | None) -> str:
    """드라이버 버전 문자열의 major 브랜치 번호로 CUDA 휠 태그를 고른다.

    파싱에 실패하거나(빈 값, 숫자가 아닌 첫 자리) 표의 어떤 구간에도 못 들면
    기본값으로 떨어진다 — #82 제안 그대로 "매핑에 없으면 지금처럼 기본값".
    """
    if not driver_version:
        return _DEFAULT_TORCH_CUDA_TAG
    try:
        major = int(driver_version.split(".", 1)[0])
    except ValueError:
        return _DEFAULT_TORCH_CUDA_TAG
    for min_major, tag in _TORCH_CUDA_TAG_BY_MIN_DRIVER_MAJOR:
        if major >= min_major:
            return tag
    return _DEFAULT_TORCH_CUDA_TAG


#: GeForce RTX 50 시리즈(Blackwell, compute capability sm_120)만 잡는다 — 대소문자
#: 무시, "RTX 50" 뒤 두 자리(5050/5060/5070/5080/5090)와 "Ti"/"Super"/"Laptop GPU"
#: 등 뒤에 붙는 접미사는 경계 문자(공백 등)만 있으면 함께 매칭된다.
#: B100/B200/GB200/RTX PRO 6000 Blackwell 같은 데이터센터·프로 카드는 명명 규칙을
#: 검증하지 않아 의도적으로 범위 밖이다(#102, ADR-0009).
_BLACKWELL_GEFORCE_NAME_RE = re.compile(r"\bRTX 50\d{2}\b", re.IGNORECASE)

#: sm_120(Blackwell) 커널은 PyTorch 2.7.0이 최초로 cu128 휠에 넣었다 — cu124·cu126은
#: 드라이버를 아무리 올려도 "no kernel image is available for execution on the
#: device"가 그대로 난다(#102). `_TORCH_CUDA_TAG_BY_MIN_DRIVER_MAJOR`에는 cu128을
#: 넣지 않았다(ADR-0007 — 특정 torch 버전에 동결돼 최신성 기준으로는 손해라서)는
#: 판단이 아키텍처 정확성보다 우선할 수 없어, 여기서만 별도로 강제한다.
_BLACKWELL_MIN_TORCH_CUDA_TAG = "cu128"


def _is_blackwell_geforce(gpu_name: str | None) -> bool:
    """GPU 이름이 GeForce RTX 50 시리즈(Blackwell)인지 판별한다."""
    return bool(gpu_name) and _BLACKWELL_GEFORCE_NAME_RE.search(gpu_name) is not None


def _torch_cuda_tag_for_env(env: dict) -> str:
    """드라이버 major 브랜치로 태그를 고른 뒤, Blackwell GeForce면 최소 cu128을 보장한다.

    cu130(드라이버 major>=580)은 cu128보다 신규 CUDA라 이미 sm_120을 포함하므로
    그대로 둔다 — override는 cu124/cu126로 떨어졌을 때만 cu128로 끌어올린다.
    """
    tag = _torch_cuda_tag_for_driver(env.get("gpu_driver_version"))
    if tag in (_DEFAULT_TORCH_CUDA_TAG, "cu126") and _is_blackwell_geforce(env.get("gpu_name")):
        return _BLACKWELL_MIN_TORCH_CUDA_TAG
    return tag


_FIX_MAP: dict[str, tuple[str, list[str] | None]] = {
    "bnb_not_compiled_with_cuda": (
        "bitsandbytes가 CUDA 지원 없이 빌드됨",
        ["-m", "pip", "install", "bitsandbytes", "--upgrade", "--force-reinstall"],
    ),
    "torch_not_installed": (
        (
            "PyTorch가 설치되어 있지 않습니다 — NVIDIA GPU가 감지됐으므로 드라이버에"
            " 맞는 CUDA 빌드 torch를 설치하면 GPU를 쓸 수 있습니다"
        ),
        # 실제 args는 suggest_fix()가 env.gpu_driver_version을 보고 동적으로 만든다
        # (#82, ADR-0007) — torch_cpu_only_build와 같은 기계를 쓴다. 여기 값은
        # 자리만 차지하며 쓰이지 않는다.
        #
        # 맨몸 `pip install torch`는 CPU 전용 빌드를 설치해 "고쳤다고 생각했는데 여전히
        # CPU"라는 더 나쁜 상태를 만든다(#55 실측). `--index-url`로 휠을 지정하기
        # 때문에 그 함정을 피한다.
        None,
    ),
    "torch_not_installed_no_gpu": (
        (
            "PyTorch가 설치되어 있지 않고 NVIDIA GPU도 조회되지 않았습니다 —"
            " GPU 기계라면 드라이버를 먼저 확인하고, 그 뒤"
            " https://pytorch.org/get-started/locally/ 에서 환경에 맞는 설치 명령을"
            " 확인하세요"
        ),
        # GPU가 보이지 않는 상태에서 2GB가 넘는 CUDA 빌드를 자동으로 받아봐야 달라지는
        # 게 없다 — torch_cpu_only_build_no_gpu와 같은 판단이다. 안내만 하고 --yes
        # 대상에서 뺀다.
        None,
    ),
    "transformers_not_installed": (
        (
            "transformers가 설치되어 있지 않습니다 — --model 체크는 이 라이브러리로"
            " 모델 config를 조회하므로 설치해야 진단할 수 있습니다: pip install transformers"
        ),
        # torch와 달리 빌드 변종이 없어 명령이 단순하지만, transformers는 torch 버전과
        # 호환 범위가 얽혀 있어 자동 설치가 기존 환경을 흔들 수 있다. 안내만 한다.
        None,
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
    "torch_cpu_only_build": (
        (
            "설치된 torch가 CPU 전용 빌드다 (torch.version.cuda 없음) — NVIDIA GPU는"
            " 감지됐으므로 CUDA 빌드 torch로 재설치하면 GPU를 쓸 수 있다"
        ),
        # fix_argv는 suggest_fix()가 env.gpu_driver_version을 보고 동적으로 만든다(#82) —
        # 여기 args는 자리만 차지하며 쓰이지 않는다.
        None,
    ),
    "torch_cpu_only_build_no_gpu": (
        (
            "설치된 torch가 CPU 전용 빌드이고 (torch.version.cuda 없음) NVIDIA GPU도"
            " 조회되지 않았다 — GPU 기계라면 드라이버 확인 후 CUDA 빌드 torch 재설치가"
            f" 필요하다: {_TORCH_CUDA_REINSTALL_DISPLAY}"
        ),
        # GPU가 실제로 보이지 않는 상태에서 2GB가 넘는 CUDA 빌드를 자동으로 받아봐야
        # 달라지는 게 없다 — 명령은 위 문구로 안내만 하고 --yes 대상에서 뺀다.
        None,
    ),
    "no_nvidia_gpu_or_driver": (
        (
            "torch는 CUDA 빌드인데 NVIDIA GPU/드라이버를 찾지 못했다 (드라이버 미설치,"
            " GPU 없음, NVML 조회 실패 중 하나) — 드라이버 상태를 먼저 확인해야 한다"
        ),
        None,
    ),
    "cuda_device_not_visible": (
        (
            "torch는 CUDA 빌드이고 GPU도 조회되는데 CUDA 장치가 이 프로세스에 보이지"
            " 않는다 (CUDA_VISIBLE_DEVICES로 가려짐, 드라이버·CUDA 런타임 불일치 등)"
            " — bitsandbytes 재설치로는 해결되지 않는다"
        ),
        None,
    ),
    "4bit_cpu_fallback_other": (
        # "CUDA 메모리 부족"이라고 단정하던 문구를 뺐다 — 여기까지 온 결과는 위
        # 원인들에 다 해당하지 않아 **원인을 특정하지 못한** 경우다(#55).
        "4bit 양자화 레이어가 CPU로 분기됨 (원인 특정 실패 — 상세 로그와 --json의 env 확인 필요)",
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
    # 실제 문구와 args는 suggest_fix()가 env를 보고 동적으로 만든다 — 없는 것만
    # 모아 pip 한 줄로 조립한다. 여기 값은 env를 못 읽었을 때의 폴백이다.
    #
    # 조합마다 cause를 만들지 않는 이유: 라이브러리가 하나 늘 때마다 조합이
    # 배로 늘어난다. cause는 하나로 두고 명령만 조립한다(torch_not_installed가
    # 드라이버를 보고 휠 태그를 고르는 것과 같은 기계다, #82·ADR-0007).
    "qlora_stack_not_installed": (
        "QLoRA 학습에 필요한 라이브러리가 설치되어 있지 않습니다",
        None,
    ),
    # 실제 문구는 suggest_fix()가 env.model_max_position과 seq_len으로 숫자를 채워
    # 만든다 — 여기 값은 그 둘을 못 읽었을 때의 폴백이다.
    #
    # fix_command는 None이다. 고칠 대상이 패키지가 아니라 **사용자가 준 인자**라
    # --yes가 자동으로 실행할 것이 없다(ADR-0008: 사용자 판단이 필요한 원인).
    "seq_len_exceeds_model_max": (
        "--seq-len이 모델의 최대 길이를 넘습니다 — 더 작은 값으로 지정하세요",
        None,
    ),
    "unknown_error": ("Canary 실행 오류 (설정 또는 환경 확인 필요)", None),
    "unknown": ("알 수 없는 이상 진단 결과", None),
}


def _cuda_visible_devices_note(value: str | None) -> str:
    """`cuda_device_not_visible` 메시지에 덧붙일 진단 정보(#81).

    fix_command는 붙이지 않는다 — 환경변수 문제라면 사용자가 직접 판단·수정할
    문제지 우리가 대신 고칠 대상이 아니다. 대신 원인 조사에 바로 쓸 수 있게 현재
    값을 보여준다.
    """
    if value is None:
        return "CUDA_VISIBLE_DEVICES는 설정되지 않음 — 드라이버/CUDA 런타임 버전 불일치 등 다른 원인일 수 있다"
    return f"현재 CUDA_VISIBLE_DEVICES={value!r}"


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
    if cause in ("torch_cpu_only_build", "torch_not_installed"):
        # 둘 다 "이 드라이버에 맞는 CUDA 빌드 torch를 받아라"로 귀결된다. 아예 없는
        # 경우에도 --force-reinstall은 무해하므로 같은 인자를 쓴다(#44).
        env = check_result.get("env") or {}
        tag = _torch_cuda_tag_for_env(env)
        args = _torch_cuda_reinstall_args(tag)
    if cause == "qlora_stack_not_installed":
        assembled = _qlora_stack_fix(check_result.get("env") or {})
        if assembled is not None:
            message, args = assembled
    if cause == "seq_len_exceeds_model_max":
        env = check_result.get("env") or {}
        note = _seq_len_note(check_result.get("seq_len"), env.get("model_max_position"))
        if note is not None:
            message = note
    if cause == "cuda_device_not_visible":
        env = check_result.get("env") or {}
        message = f"{message} ({_cuda_visible_devices_note(env.get('cuda_visible_devices'))})"
    fix_argv, fix_command = _build_command(args)
    return {
        "cause": cause,
        "message": message,
        "fix_command": fix_command,
        "fix_argv": fix_argv,
    }


#: `env` 필드 → 설치할 pip 패키지 지정자.
#: bitsandbytes는 버전 하한이 있다 — 구버전이면 transformers가 그대로 거절한다
#: ("requires bitsandbytes: pip install -U bitsandbytes>=0.46.1", 실측).
_QLORA_STACK_PACKAGES = {
    "bitsandbytes_installed": "bitsandbytes>=0.46.1",
    "peft_installed": "peft",
}


def _qlora_stack_fix(env: dict) -> tuple[str, list[str]] | None:
    """없는 라이브러리만 모아 **하나의** pip 명령으로 조립한다 (#117).

    둘 다 없을 때 명령을 나누면, 사용자가 하나 고치고 다시 돌렸다가 또 다른 게
    없다는 말을 듣는다. `--yes`도 한 번에 끝나야 한다.

    `False`(설치 안 됨 확정)인 것만 담는다 — `None`은 "못 읽었다"이므로 명령에
    넣지 않는다. 하나도 확정되지 않았으면 None을 돌려주고 호출 측이 `_FIX_MAP`의
    기본 문구를 쓰게 한다.
    """
    missing = [pkg for field, pkg in _QLORA_STACK_PACKAGES.items() if env.get(field) is False]
    if not missing:
        return None

    names = ", ".join(pkg.split(">=")[0] for pkg in missing)
    message = (
        f"{names}가 없어 QLoRA 학습을 시작할 수 없습니다"
        " — 이 상태로 시작하면 ImportError로 즉시 종료됩니다"
    )
    return message, ["-m", "pip", "install", "-U", *missing]


def _seq_len_note(seq_len, max_position) -> str | None:
    """실제 숫자를 넣은 안내. 둘 중 하나라도 못 읽으면 None(폴백 문구를 쓴다).

    **준 값과 허용 최대값을 함께 보여준다** — "너무 큽니다"만으로는 얼마로
    줄여야 하는지 모른다. 사용자가 바로 다시 실행할 수 있는 숫자를 준다(#86).
    """
    if not isinstance(seq_len, int) or not isinstance(max_position, int):
        return None
    return (
        f"seq_len({seq_len})이 모델의 최대 길이({max_position})를 넘습니다"
        f" — --seq-len을 {max_position} 이하로 지정하세요"
    )


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
