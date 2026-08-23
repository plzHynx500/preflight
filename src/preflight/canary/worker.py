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

import ctypes
import importlib.util
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

#: 즉사(SIGSEGV)하면 자식은 아무것도 쓰지 못하고, 부모는 **이 문구를 그대로**
#: `error_log`로 읽는다. 그래서 이 문구는 사용자에게 보이는 안내인 동시에 원인
#: 분류의 입력이기도 하다.
#:
#: **여기에 라이브러리 이름을 적으면 안 된다.** `fix/causes.py`가 `error_log`에서
#: 시그니처 문자열(`bitsandbytes` 등)을 찾아 원인을 정하는데, 이 문구에 그 단어가
#: 들어 있으면 **무엇이 죽었든 그 라이브러리 탓으로 분류된다** — torch의 .so가
#: 죽어도 "bitsandbytes를 재설치하라"가 나가고 `--yes`면 실제로 실행된다(#93).
#:
#: 어느 import에서 죽었는지는 문구가 아니라 `env`가 어디까지 채워졌는지로 알아야
#: 한다(#93의 후속안 A). 지금은 그 값이 없으므로 **원인을 단정하지 않는 것**이
#: 목적이다 — 틀린 답보다 "모르겠다"가 낫다.
_PREWRITE_IMPORT_NOTE = (
    "학습 스택을 import하는 도중 프로세스가 예외 없이 종료됐다. "
    "네이티브 라이브러리(.so/.dll) 로드 실패로 인한 즉사가 대표적인 경우다."
)
_PREWRITE_RUN_NOTE = (
    "canary 실행 도중 프로세스가 예외 없이 종료됐다. import는 통과한 상태였다. "
    "호스트 RAM 부족으로 인한 강제 종료, 드라이버 크래시 등이 후보다 — "
    "VRAM 부족(OOM)은 보통 예외로 잡히므로 이 경로로 오지 않는다."
)


def main() -> None:
    spec_path, result_path = sys.argv[1], sys.argv[2]

    # import를 시도하기 전에 채울 수 있는 것부터 채운다. `find_spec`은 모듈을 찾기만
    # 하고 실행하지 않으므로 여기서 불러도 안전하고, **어느 경로로 죽든 이 값은
    # 살아남는다** — 아래 세 곳의 실패 기록이 모두 이 `env`를 그대로 넘긴다(#44).
    env = _installed_env()

    # 죽기 전에 미리 기록한다. 지금부터 import를 통과하기 전까지 프로세스가 즉사하면
    # 이 결과가 그대로 부모에게 읽힌다. import로만 읽히는 속성은 아직 전부 None이지만
    # 설치 여부는 이미 담겨 있다 — 계약이 "어느 경우에도 키는 있다"고 약속한다
    # (canary-api.md).
    _write_result(result_path, _blank_result(STATUS_IMPORT_CRASH, _PREWRITE_IMPORT_NOTE, env))

    try:
        with open(spec_path, encoding="utf-8") as spec_file:
            spec = json.load(spec_file)
    except Exception:  # noqa: BLE001 - spec을 못 읽는 것도 진단 결과로 포장한다
        _write_result(result_path, _blank_result(STATUS_ERROR, traceback.format_exc(), env))
        return

    try:
        torch = _import_canary_stack()
    except BaseException:  # noqa: BLE001 - SystemExit까지 포함해 import 실패로 본다
        # 여기서 _collect_env()를 부르지 않는다 — 방금 실패한 import를 그대로 다시
        # 시도하는 꼴이라 위험만 반복된다. 대신 **이미 읽어둔 `env`를 그대로 넘긴다**
        # (#44) — `find_spec`은 import가 아니라 다시 읽는 것이 아니고, 이 값이 있어야
        # 부모가 "torch가 아예 없다"와 "있는데 로드가 깨졌다"를 가를 수 있다.
        _write_result(
            result_path,
            _blank_result(STATUS_IMPORT_CRASH, traceback.format_exc(), env),
        )
        return

    # import를 통과했다 — 여기서부터 즉사하면 더 이상 import 문제가 아니다.
    env = _collect_env()
    _write_result(result_path, _blank_result(STATUS_ERROR, _PREWRITE_RUN_NOTE, env))

    try:
        result = _run(torch, spec, env)
    except Exception as exc:  # noqa: BLE001 - 어떤 실패든 진단 결과로 포장해야 한다
        status = STATUS_OOM if _is_oom(torch, exc) else STATUS_ERROR
        result = _blank_result(status, traceback.format_exc(), env)

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


def _safe_read(reader):
    """속성 하나를 읽는다. 어떤 이유로든 실패하면 None.

    환경 속성은 **원인 분류를 돕는 부가 정보**지 진단의 전제가 아니다. 하나를 못
    읽었다고 canary 실행 자체를 멈추면, 정작 진단해야 할 환경(라이브러리가 깨진
    환경)에서 아무 결과도 못 내놓게 된다.
    """
    try:
        return reader()
    except Exception:  # noqa: BLE001 - 속성 하나를 못 읽는 건 진단 실패가 아니다
        return None


def _read_torch_version():
    import torch

    return torch.__version__


def _read_torch_cuda_version():
    import torch

    # None이면 CPU 전용 빌드 — "GPU가 없다"와 "torch가 GPU를 모른다"를 가르는 값이다.
    return torch.version.cuda


def _read_bnb_compiled_with_cuda():
    # `lib`은 모듈이 아니라 `cextension` 모듈의 **속성**이다. `import a.b.lib` 형태로는
    # ModuleNotFoundError가 나므로 from-import로 가져와야 한다.
    from bitsandbytes.cextension import lib

    return bool(lib.compiled_with_cuda)


#: `env`가 담는 키. 아무것도 못 읽는 환경에서도 이 형태는 유지된다 — 소비자가
#: 키 존재 여부까지 따로 방어하지 않아도 되게 한다.
ENV_FIELDS = (
    "torch_version",
    "torch_cuda_version",
    "bnb_compiled_with_cuda",
    # CPU 기준선을 실제로 4bit으로 잴 수 있었는지. 4bit을 시도한 경우에만 채워지고,
    # 시도조차 안 한 경우(GPU 쪽이 이미 폴백)는 None으로 남는다 — _run_basic_check 참고.
    "bnb_cpu_4bit_supported",
    # cuda_device_not_visible 원인의 진단 정보용(#81) — fix_command는 안 붙이되
    # 현재 값을 화면에 보여준다. import 없이 읽는 값이라 import 실패와 무관하게
    # 항상 채워진다(설정 안 됐으면 None).
    "cuda_visible_devices",
    # 모델 config가 말하는 최대 위치 길이. `--model` 모드에서 config를 읽은 뒤에만
    # 채워지고, 기본 체크나 속성이 없는 모델에서는 None으로 남는다.
    #
    # `seq_len`이 이 값을 넘으면 **위치 인코딩이 테이블 방식인 모델만** 죽는다
    # (#86) — 부모가 그 판정을 하려면 값이 실패 결과에도 실려 있어야 한다.
    "model_max_position",
    # QLoRA 학습에 필요한 라이브러리들의 **설치 여부**. `find_spec`으로 읽으므로
    # import를 시도하기 전에, import가 죽는 환경에서도 안전하게 채울 수 있다(#44, #92).
    #
    # 아래 `torch_version` 류와 뜻이 다르다 — 그쪽은 "import에 성공해서 읽은 값"이라
    # import 전에는 알 수 없다. 반면 이 값들은 **canary를 돌리지 않고도 참인 환경 사실**
    # 이라 `env` 기준에 정확히 들어맞는다(canary-api.md의 `env` 절).
    #
    # 읽지 못한 경우(손상된 메타데이터 등)는 None으로 남는다 — "설치 안 됨(False)"과
    # "모름(None)"을 섞지 않는다.
    "torch_installed",
    "transformers_installed",
    "bitsandbytes_installed",
    "peft_installed",
    "accelerate_installed",
)

#: `find_spec`으로 설치 여부를 확인할 라이브러리.
#:
#: **canary가 쓰는 것이 아니라 사용자의 QLoRA 학습에 필요한 것**을 넣는다 — 둘은 다르다.
#: canary는 `peft` 없이 LoRA 어댑터를 직접 만들고(model.py `_attach_manual_lora`),
#: `from_pretrained`를 안 타서 `accelerate`도 거치지 않는다. 하지만 사용자가 실제로
#: QLoRA 학습을 돌리려면 넷 다 있어야 하고, 없으면 첫 줄에서 ImportError로 죽는다.
#: **"우리가 안 쓴다"는 이유로 빼면 canary가 우회한 만큼이 그대로 진단의 사각지대가 된다**(#117).
#:
#: 여기서는 값을 `env`에 담기만 한다 — 판정·문구·수정 명령에 쓰는 것은 #117의 몫이다.
CHECKED_LIBRARIES = ("torch", "transformers", "bitsandbytes", "peft", "accelerate")


def _installed_env() -> dict:
    """import를 시도하기 **전에** 채울 수 있는 환경 사실 — 각 라이브러리의 설치 여부.

    `importlib.util.find_spec`은 모듈을 **찾기만 하고 실행하지 않는다.** 진단 대상이
    바로 "그 import가 죽는 환경"이므로(ADR-0002) 이 성질이 중요하다 — import를
    시도하기 전에, 그리고 죽지 않고 확인할 수 있다.

    그래서 이 값들은 `torch_version` 같은 "import에 성공해야 읽히는 값"과 뜻이 다르다.
    후자가 `None`인 것은 "여기까지 못 갔다"일 수도 있지만, 이쪽은 **어느 경로에서든
    같은 답**이라 진행도 신호와 섞이지 않는다.

    `find_spec` 자체가 실패할 수도 있다(손상된 메타데이터, 이상한 finder 등). 그때는
    `None`으로 남긴다 — **"설치 안 됨(False)"과 "모름(None)"을 섞지 않는다.**
    """
    env = _empty_env()
    for module_name in CHECKED_LIBRARIES:
        env[f"{module_name}_installed"] = _safe_read(
            lambda name=module_name: importlib.util.find_spec(name) is not None
        )
    return env


def _empty_env() -> dict:
    """아무 속성도 읽을 수 없을 때의 골격 — 키는 유지하고 값만 전부 None이다.

    `import torch`가 이미 실패한 상황에서 쓴다. 거기서 `_collect_env()`를 부르면
    **방금 죽은 import를 그대로 다시 시도**하는 꼴인데, 실패한 import는 캐시되지
    않아 모듈이 재실행되고 같은 이유로 또 실패한다 — 얻는 것 없이 위험만 반복한다.
    """
    return dict.fromkeys(ENV_FIELDS)


def _collect_env() -> dict:
    """원인 분류용 환경 속성 (docs/contracts/canary-api.md의 `env`).

    **부모가 아니라 자식이 읽는다.** 이 값들을 읽으려면 torch·bitsandbytes를
    import해야 하는데, 진단 대상이 바로 "그 import가 죽는 환경"이다. 부모가 읽으면
    원인을 확인하려다 CLI까지 함께 죽어 FR-03 격리가 무너진다 (Issue #19).

    항목별로 독립적으로 실패할 수 있고, 실패한 항목만 None이 된다.
    """
    env = _installed_env()
    env["torch_version"] = _safe_read(_read_torch_version)
    env["torch_cuda_version"] = _safe_read(_read_torch_cuda_version)
    env["bnb_compiled_with_cuda"] = _safe_read(_read_bnb_compiled_with_cuda)
    # os.environ 조회는 import가 아니라 실패할 수 없다 — _safe_read로 감쌀 이유가 없다.
    env["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return env


def _read_rss_peak_mb():
    """이 프로세스가 지금까지 물리 RAM에 올려둔 **최고점**(MB). 못 재면 None.

    현재값이 아니라 최고점을 쓴다. 이 값을 보는 시점은 사후(프로세스가 죽은 뒤)라
    "지금 얼마나 쓰고 있나"는 의미가 없고, **"여태 얼마까지 썼었나"** 가 원인을
    좁힌다. 커널이 최고점을 대신 기록해주므로 우리가 주기적으로 샘플링할 필요도 없다
    — torch import처럼 잠깐 치솟았다 내려가는 구간도 그대로 남는다.

    의존성을 늘리지 않으려고 OS API를 직접 부른다(`psutil` 없이). macOS 등 아래 두
    경로가 없는 플랫폼에서는 None이며, MVP 지원 대상은 Linux·Windows다(NFR-01).

    **실패를 0으로 적지 않는다** — 0은 "RAM을 안 썼다"는 뜻이 되어, 사후에 사인을
    좁히려고 이 값을 볼 때 정반대 결론으로 이끈다.
    """
    if sys.platform == "win32":
        return _safe_read(_read_rss_peak_mb_windows)
    return _safe_read(_read_rss_peak_mb_proc)


class _ProcessMemoryCounters(ctypes.Structure):
    """Windows PROCESS_MEMORY_COUNTERS. `PeakWorkingSetSize`가 RSS 최고점이다."""

    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _read_rss_peak_mb_windows():
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)

    # restype을 지정하지 않으면 ctypes가 반환값을 int(32비트)로 해석해 64비트 핸들이
    # 잘린다. 그러면 조회가 조용히 실패해 0이 나온다 — 반드시 포인터 크기로 받는다.
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_ProcessMemoryCounters),
        ctypes.c_uint32,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int

    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo 실패")
    return counters.PeakWorkingSetSize / (1024 * 1024)


def _read_rss_peak_mb_proc():
    # /proc/self/status의 VmHWM이 resident set의 최고점(high water mark)이다.
    # /proc/self/statm에는 현재값만 있고 최고점이 없어 이쪽을 쓴다.
    with open("/proc/self/status", encoding="ascii") as status:
        for line in status:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024  # kB → MB
    raise OSError("/proc/self/status에 VmHWM 항목이 없다")


def _blank_result(status: str, error_log: str | None, env: dict | None = None) -> dict:
    """측정값이 없는 실패 결과. 계약 스키마(docs/contracts/canary-api.md)를 채운다.

    `rss_peak_mb`는 **호출 시점에** 잰다. 사전 기록의 목적이 죽었을 때 상태를 남기는
    것이라, 마지막 한 번만 재면 정작 죽었을 때의 값이 남지 않는다 (Issue #27).
    최고점이라 이전 단계의 급증도 함께 실린다.

    `env`는 인자로 받는다 — import 전 사전 기록에서 이 함수가 직접 수집하면 그
    수집 과정(`import torch`)에서 죽어버려 사전 기록 자체가 불가능해진다.
    """
    return {
        "status": status,
        "device": None,
        "memory_delta_mb": None,
        "elapsed_ms": None,
        "cpu_multiplier": None,
        "quant_backend": None,
        "error_log": error_log,
        "env": env,
        "rss_peak_mb": _read_rss_peak_mb(),
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


def _run(torch, spec: dict, env: dict) -> dict:
    model_name = spec.get("model_name")
    raw_batch_size = spec.get("batch_size")
    raw_seq_len = spec.get("seq_len")

    # CLI는 `min=1`로 막지만(#59), run_canary_check()를 직접 호출하는 경로(CLI 밖,
    # 향후 API)는 그 방어를 거치지 않는다. 여기서 막지 않으면 0·음수가 그대로
    # torch.randint(..., (0, 8)) 같은 빈/유효하지 않은 텐서로 흘러가 forward+backward가
    # 연산 없이 "성공"해버려 status="ok"(PASS)가 나간다 — 아무것도 안 됐는데 초록불이
    # 켜지는 거짓 양성이다. 명시적으로 준 값만 검사한다 — None(미지정)은 기본값을 쓰는
    # 정상 경로이므로 걸리면 안 된다.
    if (raw_batch_size is not None and int(raw_batch_size) < 1) or (
        raw_seq_len is not None and int(raw_seq_len) < 1
    ):
        return _blank_result(
            STATUS_ERROR,
            f"batch_size/seq_len은 1 이상이어야 한다 "
            f"(batch_size={raw_batch_size}, seq_len={raw_seq_len})",
            env,
        )

    batch_size = int(raw_batch_size or DEFAULT_BATCH_SIZE)
    seq_len = int(raw_seq_len or DEFAULT_SEQ_LEN)

    if model_name is None:
        return _run_basic_check(torch, batch_size, seq_len, env)

    return _run_model_check(torch, model_name, batch_size, seq_len, env)


#: 모델 config에서 최대 위치 길이를 담는 속성 이름 — 앞에서부터 시도한다(#86).
#:
#: transformers 5.14.1 의 `configuration_*.py` 482개를 실제로 세어본 결과다:
#:
#:   max_position_embeddings   257개   현행 표준. 대부분 여기서 끝난다
#:   n_positions                 8개   GPT-2 계열
#:   max_seq_len                 2개   MPT · DBRX
#:
#: GPT-2 는 config 가 실제로는 `n_positions` 에 담지만 `attribute_map` 으로 별칭이
#: 걸려 있어(`configuration_gpt2.py:73`, 이런 config 가 18개) 첫 이름으로도 읽힌다 —
#: #86 재현에서 tiny-random-gpt2 가 512로 읽힌 것이 그 증거다. `n_positions` 는 그
#: 별칭이 없는 커스텀 config 를 위한 방어다.
#:
#: **셋 다 없는 config 도 많다**(비전·오디오 타워 등 텍스트 위치 한계가 없는 것들).
#: 그때는 None 이고, 받는 쪽은 단정하지 않는다 — 아래 docstring 참고.
_MAX_POSITION_ATTRS = ("max_position_embeddings", "n_positions", "max_seq_len")


def _max_position_len(config) -> int | None:
    """모델 config가 말하는 최대 위치 길이. 못 읽으면 None.

    **"없음"을 "제한 없음"으로 읽지 않는다.** 속성이 하나도 없으면 그 모델의
    한계를 모르는 것이지 한계가 없는 것이 아니다 — None을 받은 쪽은 단정하지
    않고 기존 분류로 흘려보낸다("모르는 것"과 "아닌 것"을 섞지 않는다).
    """
    for attr in _MAX_POSITION_ATTRS:
        value = getattr(config, attr, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _run_model_check(torch, model_name: str, batch_size: int, seq_len: int, env: dict) -> dict:
    from preflight.canary.model import build_dummy_input as build_hf_dummy_input

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, config, quant_backend = build_dummy_model(model_name, device)
    # **forward보다 먼저 담는다.** seq_len 초과는 GPU 커널 안에서 터지는데, 부모가
    # 그걸 원인으로 특정하려면 이 값이 실패 결과에 실려 있어야 한다. `env`는
    # main()이 준 것과 같은 dict라 여기서 채우면 예외 경로의 기록에도 그대로 남는다.
    env["model_max_position"] = _max_position_len(config)
    dummy_input = build_hf_dummy_input(batch_size, seq_len, config.vocab_size, device)

    measured = _execute_canary_cycle(torch, model, dummy_input, device, quant_backend)

    return {
        "status": STATUS_OK,
        "device": measured["device"],
        "memory_delta_mb": measured["memory_delta_mb"],
        "elapsed_ms": measured["elapsed_ms"],
        "cpu_multiplier": None,
        "quant_backend": measured["quant_backend"],
        "error_log": None,
        "env": env,
        "rss_peak_mb": _read_rss_peak_mb(),
    }


def _run_basic_check(torch, batch_size: int, seq_len: int, env: dict) -> dict:
    """기본 체크 — 모델과 무관하게 GPU/드라이버/CUDA 체인이 살아있는지 실측한다."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # CPU는 float16 연산 유닛이 사실상 없어 정상 경로인 float32로 돌린다
    # (docs/architecture.md §6-01 "dtype 비대칭" 참고).
    dtype = torch.float16 if device == "cuda" else torch.float32

    measured = _measure(torch, device, dtype, batch_size, seq_len)

    cpu_multiplier = None
    if device == "cuda" and measured["elapsed_ms"]:
        tried_4bit = measured["quant_backend"] == QUANT_BACKEND_4BIT
        baseline_ms, baseline_backend = _measure_cpu_baseline(
            torch, batch_size, seq_len, tried_4bit
        )
        if baseline_ms is not None:
            cpu_multiplier = baseline_ms / measured["elapsed_ms"]
        # 4bit을 실제로 시도했을 때만 지원 여부를 말할 수 있다. GPU 쪽이 이미 폴백해
        # CPU에서도 시도조차 안 했다면 "지원 안 됨"이 아니라 "모름"이므로 None으로 둔다.
        if tried_4bit and baseline_backend is not None:
            env["bnb_cpu_4bit_supported"] = baseline_backend == QUANT_BACKEND_4BIT

    return {
        "status": STATUS_OK,
        "device": measured["device"],
        "memory_delta_mb": measured["memory_delta_mb"],
        "elapsed_ms": measured["elapsed_ms"],
        "cpu_multiplier": cpu_multiplier,
        "quant_backend": measured["quant_backend"],
        "error_log": None,
        "env": env,
        "rss_peak_mb": _read_rss_peak_mb(),
    }


def _measure_cpu_baseline(torch, batch_size: int, seq_len: int, prefer_4bit: bool):
    """CPU 강제 폴백의 (실행 시간 ms, 실제로 쓰인 quant_backend).

    절대 시간은 GPU 세대마다 크게 달라 쓸 수 없으므로, 이 값 대비 몇 배 빠른가로
    판정한다 (docs/adr/0003-relative-baseline-timing.md). 측정에 실패하면 (None, None).

    backend를 함께 돌려주는 이유는, CPU 4bit이 `nn.Linear`보다 5.3배 느려서
    (docs/architecture.md §6-01) 양쪽 backend가 어긋나면 배수 자체가 왜곡되기
    때문이다 — 그 사실을 `env`에 남겨 원인 분류가 참고할 수 있게 한다.
    """
    try:
        baseline = _measure(
            torch, "cpu", torch.float32, batch_size, seq_len, prefer_4bit=prefer_4bit
        )
    except Exception:  # noqa: BLE001 - 기준선 실패가 canary 실패로 번지면 안 된다
        # 기준선을 못 재는 것은 canary 자체의 실패가 아니다 — 배수만 포기한다.
        return None, None
    return baseline["elapsed_ms"], baseline["quant_backend"]


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
    return _execute_canary_cycle(torch, model, dummy_input, device, quant_backend)


def _execute_canary_cycle(torch, model, dummy_input, device: str, quant_backend: str):
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
    outputs = model(dummy_input)
    if hasattr(outputs, "logits"):
        loss = outputs.logits.float().pow(2).mean()
    else:
        loss = outputs.float().pow(2).mean()
    loss.backward()
    optimizer.step()


def _base_layer_device(model):
    """4bit 베이스 레이어 파라미터가 실제로 올라가 있는 device.

    "4bit 레이어 device=cpu 감지"가 판정 항목이므로 모델 전체가 아니라 베이스
    레이어를 직접 본다.

    얼린 파라미터가 하나도 없으면 **첫 파라미터의 device**로 물러선다. 원래대로면
    그런 모델에서 None이 나가, 화면에 `device=None`이 찍히고 진짜 CPU에 있어도
    judge의 `device=="cpu"` FAIL 규칙이 발동하지 못했다(#66).

    이 물러섬을 만든 원인이던 `--model` 경로의 폴백 모델은 #75에서 베이스를 얼리게
    되어 이제 정상적으로 얼린 베이스를 찾는다. 그래도 물러섬은 남겨둔다 — 폴백에서
    LoRA를 붙일 대상 `nn.Linear`가 하나도 없거나 `import torch`가 죽어 동결을
    되돌리는 경로가 여전히 있어서, "얼린 파라미터가 없는 모델"이 사라진 것은 아니다.
    파라미터가 아예 없는 모델만 None이다.
    """
    first_device = None
    for param in model.parameters():
        if not param.requires_grad:
            return str(param.device).split(":")[0]
        if first_device is None:
            first_device = str(param.device).split(":")[0]
    return first_device


if __name__ == "__main__":
    main()
