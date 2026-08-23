# Canary 내부 API 계약

CanaryEngine / FixExecutor / ReVerifier 세 모듈이 병렬로 개발되기 위한 함수 경계다. 각 함수의 입출력 스키마를 코드를 짜기 전에 먼저 합의해둔 것이므로, 시그니처를 바꿀 때는 이 문서를 함께 갱신한다.

## `run_canary_check`

Canary를 격리된 subprocess에서 실행하고 원시 측정값을 돌려준다.

```python
def run_canary_check(model_name: str | None, batch_size: int, seq_len: int) -> dict:
    """
    Canary를 격리된 subprocess에서 실행한다.
    OOM·import 크래시까지 전부 이 함수 안에서 잡아 아래 스키마로
    정규화해 반환한다 — 절대 예외를 던지거나 프로세스를 죽이지 않는다.
    """
    return {
        "status": "ok",             # "ok" | "oom" | "import_crash" | "error"
        "device": "cuda",           # 실제 파라미터가 있던 device
        "memory_delta_mb": 5000.0,  # status != "ok"면 None
        "elapsed_ms": 12.3,         # status != "ok"면 None
        "cpu_multiplier": None,     # 기본 체크에서만 값이 들어감
        "quant_backend": "bnb-4bit",  # "bnb-4bit" | "nn-linear-fallback"
        "error_log": None,          # status != "ok"일 때만 채워짐 — 원본 stderr/예외 메시지
        "env": {...},               # 환경 사실 — 아래 참고. 못 읽으면 None
        "rss_peak_mb": 1401.7,      # 이 기록을 남긴 시점까지의 호스트 RAM 최고점. 못 재면 None
    }
```

**`batch_size`·`seq_len`은 1 이상이어야 한다.** 0·음수를 명시적으로 주면 canary를 실행하지 않고 `status="error"`로 즉시 반환한다(#59) — 막지 않으면 `torch.randint(..., (0, 8))` 같은 빈/유효하지 않은 텐서로 forward+backward가 연산 없이 "성공"해버려, 아무것도 측정하지 않았는데 `status="ok"`가 나가는 거짓 양성이 된다. CLI는 파싱 단계(`min=1`)에서 먼저 막지만, 이 함수를 직접 호출하는 경로는 그 방어를 거치지 않으므로 함수 자체도 값을 검증한다. `None`(미지정)은 기본값(`batch_size=1`, `seq_len=8`)을 쓰는 정상 경로라 걸리지 않는다.

**첫 인자는 모델 객체가 아니라 모델명이다.** 부모가 `nn.Module`을 만들어 넘기려면 부모 프로세스가 `torch`를 import해야 하는데, `import torch`·`import bitsandbytes` 자체가 죽는 상황을 잡는 것이 프로세스 격리([ADR-0002](../adr/0002-subprocess-isolation-for-canary.md))의 목적이므로 그 순간 격리가 무력화된다. 따라서 모델 구성은 전부 자식 프로세스 안에서 일어난다 — 모델을 만드는 코드(`canary/model.py`)도 자식에서만 호출되므로 함수 안에서만 torch/transformers를 import해야 하고, 부모 쪽 모듈에 import를 노출하면 안 된다. `model_name`이 `None`이면 기본 체크(모델과 무관한 최소 대표 구조)를 뜻한다.

`status`는 사람이 읽는 값이 아니라 다음 단계가 기계적으로 분기하는 스위치다. `"import_crash"`·`"oom"`은 각각 재설치·batch 축소라는 서로 다른 FIX로 이어지므로, 그 둘로 확정할 수 없는 실패(모델명 오타로 config 조회 실패, 미구현 경로 호출 등)를 둘 중 하나로 적으면 사용자에게 엉뚱한 해결 명령이 제시된다. 그런 실패는 `"error"`로 두고 `error_log`에 원본 메시지를 담는다.

`quant_backend`는 canary가 실제로 어떤 연산 경로로 돌았는지를 나타낸다. 구버전 bitsandbytes 등으로 4bit 레이어 구성·실행이 실패해 평범한 `nn.Linear`로 대체된 경우 `"nn-linear-fallback"`이 되며([architecture.md §6-01](../architecture.md)), 이때는 "4bit 레이어 device=cpu" 판정을 할 수 없으므로 리포트에 폴백 사실을 명시해야 한다.

| 필드 | 채우는 쪽 | 쓰는 쪽 |
|---|---|---|
| `model_name`, `batch_size`, `seq_len` (입력) | CLI 진입점 — 모델명(또는 기본 체크면 `None`)만 넘긴다 | 실행·측정 담당(자식 프로세스에서 모델 구성) |
| `status`/`device`/`memory_delta_mb`/`elapsed_ms`/`cpu_multiplier`/`quant_backend`/`error_log` (출력) | 실행·측정·크래시 캐치까지 전부 담당 | 판정(`judge_result`) → 원인분류(`suggest_fix`) |
| `env` (출력) | 자식이 먼저 채우고, **부모가 뒤에 얹는다** — 아래 참고 | 원인분류(`suggest_fix`) · 판정(`judge_result`) · 리포트 |
| `rss_peak_mb` (출력) | 자식이 사전 기록마다 잰다 | 원인분류(`suggest_fix`)만 — **아직 아무도 안 읽는다**(#27). 판정·리포트는 앞으로도 읽지 않는다 ([ADR-0006](../adr/0006-ram-recorded-internally-only.md)) |

### `env` — 환경 사실

측정값과 달리 **canary를 돌리지 않고도 참인 값**을 담는다. 무엇을 여기 넣고 무엇을 최상위에 둘지는 한 줄 기준을 따른다(2026-08-06 회의 안건 2).

> **canary를 돌리지 않고도 알 수 있으면 `env`, 돌려야만 알 수 있으면 최상위.**

```python
"env": {
    "torch_version": "2.11.0+cu128",
    "torch_cuda_version": "12.8",       # None이면 torch가 CPU 전용 빌드
    "bnb_compiled_with_cuda": True,     # 빌드 속성이 아니다 — 아래 경고 참고
    "bnb_cpu_4bit_supported": True,     # 4bit을 실제로 시도한 경우에만. 아니면 None
    "cuda_visible_devices": None,       # os.environ["CUDA_VISIBLE_DEVICES"] 원문. 없으면 None
    "torch_installed": True,            # find_spec 기반 — import 전에도 채워진다
    "transformers_installed": True,
    "bitsandbytes_installed": False,    # False면 "설치 안 됨" 확정. None은 "못 읽음"
}
```

> **`bnb_compiled_with_cuda`는 이름과 달리 빌드 속성이 아니다 (#72, 2026-08-22 실측).** `bitsandbytes.cextension.lib.compiled_with_cuda`는 bitsandbytes 0.50 기준으로 **런타임에 CUDA 장치가 보이는지**에 따라 값이 바뀐다 — CUDA torch·bitsandbytes가 모두 정상인 RTX 4070 Ti에서 `CUDA_VISIBLE_DEVICES=-1`만 붙이면 같은 설치본이 `True` → `False`가 된다. 즉 이 값이 `False`라고 해서 "CUDA 없이 빌드된 bitsandbytes"라고 단정할 수 없고, 재설치를 권하면 아무것도 고쳐지지 않는다.
>
> 그래서 **원인 분류는 이 값을 단독 근거로 쓰지 않는다.** `torch_cuda_version`(torch가 CUDA 빌드인가)과 `gpu_free_mb`(NVML이 GPU를 조회했는가)를 먼저 보고, 그 둘을 모두 못 읽었을 때의 마지막 폴백으로만 참고한다([`suggest_fix`](#suggest_fix) 절 참고). 필드 이름과 수집 방식은 하위 호환을 위해 유지한다 — 바뀌는 것은 **읽는 쪽의 해석**이다.
>
> 구버전 bitsandbytes에서는 진짜 빌드 속성이었을 가능성이 있다. 그렇다면 이 값은 **버전에 따라 뜻이 달라지는 신호**라는 뜻이므로, 단독 판단 근거로 쓰지 않는다는 결론은 어느 쪽이든 같다.

**자식이 읽어야 한다.** 이 값들을 읽으려면 `torch`·`bitsandbytes`를 import해야 하는데, 진단 대상이 바로 *"그 import가 죽는 환경"* 이다. 부모가 읽으면 원인을 확인하려다 CLI까지 함께 죽어 FR-03 격리가 무너진다([ADR-0002](../adr/0002-subprocess-isolation-for-canary.md), Issue #19).

> `cuda_visible_devices`만 예외다 — `os.environ.get()`은 import가 아니므로 실패할 수 없고, import 성패와 무관하게 항상 값(또는 `None`)이 채워진다. 그래도 자식이 채우는 이유는 그 값을 봐야 할 순간이 정확히 "자식이 본 CUDA 장치 가시성"이라서다(ADR-0008).

항목은 **독립적으로 실패**할 수 있고, 실패한 항목만 `None`이 된다. 키는 **어느 경우에도 유지된다** — 소비자가 키 존재 여부까지 따로 방어하지 않게 하기 위함이다.

**`*_installed`는 나머지와 채워지는 시점이 다르다.** `importlib.util.find_spec`은 모듈을 **찾기만 하고 실행하지 않으므로**, import를 시도하기 **전에** 그리고 import가 죽는 환경에서도 안전하게 읽을 수 있다(#44, #92). 그래서 `status == "import_crash"`인 결과에도 이 값들은 채워져 있다 — 자식이 죽기 전에 이미 담아 보냈기 때문이다.

반면 `torch_version`·`torch_cuda_version`·`bnb_*`는 **import에 성공해야 읽히는 값**이라, import가 깨진 경우 전부 `None`이다. **이때 속성을 다시 읽으려 시도하지는 않는다**(방금 실패한 import를 반복할 뿐이다) — `find_spec`은 그 "다시 읽기"에 해당하지 않는다.

> **`*_installed`의 `False`와 `None`을 섞지 말 것.** `False`는 "설치 안 됨"이 **확정**된 것이고, `None`은 `find_spec` 자체가 실패해 **"모른다"** 는 뜻이다. 원인 분류는 `False`일 때만 단정한다.

`env` 자체를 못 받는 경우(구버전 canary 등)는 `None`이므로, 소비자는 `(raw.get("env") or {}).get(...)` 형태로 읽어야 한다.

> **`env`는 여럿이 쌓아가는 칸이다 — 통째로 갈아끼우지 말 것.** 자식이 라이브러리 상태를 채운 뒤, 부모(`cli`)가 GPU 정보를 **얹는다**. `raw["env"] = {...}`로 대입하면 자식이 채운 값이 사라지고, **에러 없이 원인 분류만 조용히 실패**해 잡기 어렵다.
>
> ```python
> raw["env"] = {**(raw.get("env") or {}), "gpu_free_mb": ..., "gpu_total_mb": ..., "gpu_driver_version": ...}
> ```
>
> **`setdefault("env", {})`를 쓰면 안 된다.** `_normalize()`가 모든 필드를 항상 만들기 때문에 `env`는 **키가 있고 값만 `None`인 상태**가 되는데, `setdefault`는 키 존재 여부만 보므로 `None`을 그대로 돌려준다 → `None.update(...)`로 **부모 프로세스가 죽는다** — 하필 `import_crash`처럼 `env`가 비는 바로 그 상황에서, ADR-0002가 막으려던 것과 같은 종류로. 자식이 아예 안 돌아 부모가 결과를 직접 만드는 경로(기동 실패·타임아웃·결과 파일 없음)에서도 `env`는 `None`이므로, 병합 코드는 어떤 경우에도 `None`을 견뎌야 한다.
>
> (이 규칙 자체가 PR #26 리뷰 중 실측으로 정정된 것이다 — 처음 문서에 적혀 있던 `setdefault().update()`가 실제로 부모를 죽인다는 게 그때 확인됐다.)

### `rss_peak_mb` — 그 시점까지의 호스트 RAM 최고점

`status == "error"`의 원인 후보를 좁히기 위한 **내부 근거**다. `verdict`에 영향을 주지 않고 화면에도 찍지 않는다 — 다만 **원인 분류(`suggest_fix`)에는 쓸 수 있다.** 이유는 [ADR-0006](../adr/0006-ram-recorded-internally-only.md) 참고.

자식은 [사전 기록](../adr/0005-pre-written-result-over-exit-code.md)을 남길 때마다 이 값을 다시 잰다. 최종 결과에만 넣으면 정작 예외 없이 즉사했을 때 아무것도 남지 않는다.

**현재값이 아니라 최고점**(Windows `PeakWorkingSetSize` · Linux `VmHWM`)이다. 이 값을 읽는 시점은 언제나 사후라 *"지금 얼마나 쓰고 있나"* 는 의미가 없고, 커널이 최고점을 대신 기록해주므로 우리가 주기적으로 샘플링하지 않아도 torch import처럼 잠깐 치솟았다 내려가는 구간이 그대로 남는다.

> **한계 — 실행 단계 도중의 증가는 못 잡는다.** 사전 기록은 각 단계에 **진입하기 전에** 쓰이므로, 실행 중 RAM이 급증한 뒤 SIGKILL로 죽으면 살아남는 값은 그 직전 단계까지의 최고점이다. 죽기 직전까지 추적하려면 부모가 자식을 폴링해야 하는데(`subprocess.run` → `Popen` + 루프) MVP 범위 밖이다 — **그때도 이 필드의 이름·의미는 그대로고 누가 재느냐만 바뀐다**(#27).
>
> 같은 이유로 **`suggest_fix`가 아직 이 값을 읽지 않는다.** *"다른 앱이 RAM을 다 써서 우리가 희생된"* 경우를 가르려면 시스템 여유 RAM이 함께 있어야 하는데, 그것도 #27에 함께 남겼다.

측정 실패 시 `0`이 아니라 `None`이다 — `0`은 *"RAM을 안 썼다"* 는 뜻이 되어 정반대 결론으로 이끈다. macOS는 두 경로가 모두 없어 항상 `None`이다(NFR-01 대상 아님).

## `build_dummy_model` / `build_dummy_input` (canary/model.py, W4 산출물)

`run_canary_check`의 자식 프로세스 안에서만 호출되는 내부 헬퍼다 — 모듈 경계 밖으로
노출되지 않지만, W9 통합(worker.py의 `--model` 경로 연결) 때 그대로 쓰일 시그니처라
여기 함께 적는다.

```python
def build_dummy_model(model_name: str, device: str = "cuda"):
    """가중치 다운로드 없이 config만 조회해 QLoRA(4bit+LoRA) 랜덤 초기화 모델을 만든다."""
    ...
    return model, config, quant_backend
    # config.vocab_size를 build_dummy_input에 넘기기 위해 함께 반환.
    # quant_backend: "bnb-4bit" | "nn-linear-fallback" (run_canary_check 반환
    # 스키마의 quant_backend와 동일한 값 — W9에서 그대로 채워 넣으면 된다)

def build_dummy_input(batch_size: int, seq_len: int, vocab_size: int):
    """토큰 ID(정수) 텐서. vocab_size가 있어야 유효한 범위의 ID를 만들 수 있다."""
    ...
```

`build_dummy_input`은 원래 `(batch_size, seq_len)`만 받는 시그니처로 스텁이 있었으나,
`--model` 경로의 입력은 정수 토큰 ID라 `vocab_size`(=`build_dummy_model`이 돌려준
`config.vocab_size`)가 있어야 만들 수 있어 W4 구현 중 `vocab_size` 파라미터를
추가했다. W9에서 두 함수를 이어 쓸 때는 `build_dummy_model`이 돌려준 `config`에서
`vocab_size`를 꺼내 `build_dummy_input`에 그대로 넘기면 된다.

`build_dummy_model`은 **항상 QLoRA(4bit 양자화 + LoRA 어댑터)를 적용한다** —
architecture.md §3 "학습 설정 세부 옵션"의 고정 가정이며, 옵션이 아니다(2026-08-03,
상영님 지적으로 W4 최초 구현에서 이 가정이 빠져 있던 것을 발견·수정).

**구현 방식이 한 번 더 바뀌었다(2026-08-06).** 최초 시도(`AutoModelForCausalLM.
from_config(config, quantization_config=...)`)는 동작하지 않는다 — 4bit 양자화는
"가중치 파일을 읽으면서" 일어나는 기능이라(`from_pretrained`가 디스크에서 한 층씩
읽으며 즉시 압축) 파일을 안 읽는 `from_config`에는 그 인자 자체가 없다. 게다가 `peft`도
하드 의존성이 아니라 없을 수 있다. 두 문제 다 `except Exception`에 조용히 삼켜져서
`quant_backend`가 환경과 무관하게 **항상** `"nn-linear-fallback"`으로 떨어지는 채로
있었다(상영님이 실물 라이브러리로 실측 발견, PR #12 리뷰 코멘트 참고).

**현재 방식**: `torch.device("meta")` 위에서 골격만 만들고 → `transformers.
integrations.bitsandbytes.replace_with_bnb_linear`(비공개 API, 버전 업 시 확인 필요)로
`nn.Linear`를 `Linear4bit`로 치환 → 레이어 단위로 랜덤 값을 채우며 GPU 4bit으로
실체화 → `peft` 없이 `Linear4bit`마다 forward hook으로 LoRA 어댑터(작은 `nn.Linear`
두 개)를 수동으로 붙인다. 체크포인트 없이도 레이어 하나 분량(최대 수백 MB)만 RAM에
잠깐 존재하고, 전체 모델이 fp16으로 통째로 RAM에 올라가는 순간이 없다 — 상영님이
RTX 4070 Ti·8B급 모델 기준 RAM 피크 1.75GB·VRAM 최고점 5.77GB로 실측 검증했다.
`device` 파라미터가 이 실체화 대상 디바이스를 그대로 결정한다.

4bit 구성이 실패하면(bitsandbytes 미설치·구버전, `replace_with_bnb_linear`가 향후
transformers 버전에서 사라지는 경우 등) `build_minimal_canary_model`과 동일한 폴백
철학으로 fp32 전체 모델로 대체하고 `quant_backend="nn-linear-fallback"`으로 알린다.

**폴백 모델도 `device`가 가리키는 곳으로 옮긴다** — 4bit 경로든 폴백 경로든 `device`
파라미터의 의미는 같다(#66). 이걸 빼먹으면 폴백 모델만 CPU에 남아, 호출자가 같은
`device`로 만든 입력(cuda)과 어긋나 `RuntimeError: Expected all tensors to be on the
same device`로 죽는다 — 폴백은 "bitsandbytes가 없어도 진단은 계속한다"는 안전장치인데
발동하는 순간 도구가 부서지므로 안전장치가 없는 것과 같아진다. GPU에 두는 대신 CPU에
두는 대안도 검토했으나, 그러면 `device=cpu`가 그대로 FAIL 판정("GPU가 죽었다")이 되어
더 나쁜 오진을 내고 `--model`의 존재 이유인 VRAM 실측도 통째로 잃는다.

단 이 폴백은 fp32 전체 모델이 RAM/VRAM에 통째로 올라가므로(8B 기준 약 30GB, 게다가
베이스를 얼리지 않아 gradient·옵티마이저 상태까지 붙는다) canary 도구 자체가 진단을
마치기 전에 먼저 죽을 수 있다 — "4bit이 안 되는 환경"이라는 신호로는 유효하지만
쾌적하게 죽지는 않는다(개선 여지, 지금 범위 밖 — #75).

폴백 모델은 베이스를 얼리지도 LoRA를 붙이지도 않아 **모든 파라미터가
`requires_grad=True`** 다. `worker._base_layer_device()`가 "얼린 베이스 레이어"를 찾아
`device` 필드를 채우므로, 얼린 파라미터가 하나도 없으면 첫 파라미터의 device로 물러선다 —
그러지 않으면 이 경로에서만 `device`가 `None`이 되어 화면에 `device=None`이 찍히고
`judge_result`의 `device=="cpu"` FAIL 규칙도 발동하지 못한다(#66).

4bit 가중치는 uint8로 packed되어 `numel()` 기준 파라미터 수가 실제의 절반으로 보인다
— 리포트에 파라미터 수를 찍을 일이 있으면 `config`에서 계산해야 한다.

## `query_gpu_state` (gpu.py, W-gpu 산출물)

`judge_result`가 VRAM 헤드룸 기반 WARN을 판정하려면 "canary를 돌리기 직전, 이 GPU에
얼마나 여유가 있었는지"가 필요하다. 이 값은 canary(자식 프로세스)를 실행해야만
알 수 있는 게 아니라 — NVML로 그냥 조회하면 되는 값이라 **canary와 무관하게 부모
프로세스에서** 얻는다(2026-08-06 회의 안건 3-1). `torch`를 import하지 않는
`pynvml`(nvidia-ml-py)만 쓰므로 ADR-0002가 격리하려는 대상(`import torch`가
프로세스를 죽이는 것)과 무관하다.

```python
def query_gpu_state(device_index: int = 0) -> dict | None:
    return {
        "name": "NVIDIA GeForce RTX 4070 Ti",
        "total_mb": 12282.0,
        "free_mb": 11500.0,
        "driver_version": "610.62",
    }
    # 실패(NVML 미설치·드라이버 없음·GPU 없음 등)하면 예외 대신 None을 돌려준다.
```

**호출 시점이 중요하다** — canary 기동 "직전" 1회만 불러야 한다. canary가 돌기
시작한 뒤에 조회하면 `free_mb`가 canary 자신의 점유만큼 깎여 오염된다. 다중 GPU는
MVP 범위 밖이라 `device_index=0`(기본 GPU)만 본다.

**`judge_result`로 넘기는 방법**: `run_canary_check()`의 7개 필드 반환 스키마 자체에는
포함되지 않는다 — 호출한 쪽(cli.py, W12)이 `query_gpu_state()`를 별도로 불러
`env`에 `gpu_free_mb`·`gpu_total_mb`(둘 다 MB)·`gpu_driver_version` 세 값을 병합한 뒤
`judge_result()`에 넘긴다. `gpu_total_mb`는 화면(report.py `_vram_line`)이 "X.XGB / Y.YGB
가용 (총 Z.ZGB)"를 판정과 같은 숫자로 보여주는 데 쓰인다. `gpu_driver_version`은 #82가
`torch_cpu_only_build`의 `fix_command`가 받을 CUDA 휠 태그를 드라이버에 맞춰 고르려고
얹은 값이다 — `suggest_fix`가 이 값을 읽어 태그를 정한다(major 브랜치 번호만 보는
근사 매핑, [ADR-0007](../adr/0007-driver-version-based-torch-cuda-wheel-selection.md)
참고). 이 값이 없거나 매핑에 없는 값이면 기존 기본값(`cu124`)으로 떨어진다.

```python
raw["env"] = {
    **(raw.get("env") or {}),
    "gpu_free_mb": state["free_mb"],
    "gpu_total_mb": state["total_mb"],
    "gpu_driver_version": state["driver_version"],
}
```

**대입이 아니라 병합이다** — `raw["env"] = {...}`로 덮어쓰면 자식이 채운 값이 사라지고,
`raw.setdefault("env", {}).update(...)`는 `env`가 `None`일 때 부모 프로세스를 죽인다.
이유와 `env`가 `None`으로 오는 경로는 위 [`env` 절](#env--환경-사실) 참고.

`env`가 없거나 `env.gpu_free_mb`가 없으면(NVML 조회
실패 등) `judge_result`는 관련 WARN 판정을 조용히 건너뛴다 — 필수 입력이 아니다.

## `judge_result`

원시 측정값에 매직넘버 기준을 비교해 `verdict`와 `reasons`를 얹는다. **원인은 몰라도 된다** — 순수 숫자·상태 비교일 뿐이다.

```python
def judge_result(raw: dict) -> dict:
    return {
        **raw,
        "verdict": "WARN",                  # "PASS" | "WARN" | "FAIL"
        "reasons": ["memory_delta_high"],   # 걸린 조건 목록(여러 개 동시 가능)
    }
```

### 판정 기준

| 판정 | 조건 |
|---|---|
| FAIL | `status == "oom"` · `status == "import_crash"` · `status == "error"` · **`device == "cpu"`**(`quant_backend`와 무관) |
| WARN | `memory_delta_mb`가 `env.gpu_free_mb`(canary 기동 직전 가용 VRAM)의 90% 이상 · `cpu_multiplier < 2` |
| PASS | 위 조건에 전부 해당하지 않음 |

FAIL 4개·WARN 2개, 총 6개 판정 항목 전부 MVP 구현으로 팀이 확정했다(`status == "error"`는 W5 구현 중 추가). 두 숫자(90%, 2배)는 정밀 검증된 값이 아니라 매직넘버로 우선 채택한 것이며, 실측 데이터가 쌓이면 조정한다.

> **`device == "cpu"` FAIL은 `quant_backend`와 무관하게 적용된다** (2026-08-03, #18로 발견된 계약 구멍 수정). 기본 체크의 목적 자체가 "GPU/드라이버/CUDA 체인이 물리적으로 살아있는가"([architecture.md §3](../architecture.md))라서, 4bit 레이어 유무와 무관하게 device가 cpu면 그 자체로 실패다. 원래는 `quant_backend == "bnb-4bit"`일 때만 이 조건을 봤는데, 그러면 bitsandbytes 자체가 없어 4bit 레이어가 없는 환경(`quant_backend == "nn-linear-fallback"`)은 이 규칙이 발동하지 못해 GPU가 전혀 없는데도 PASS가 나가는 구멍이 있었다. reason 이름은 `"quant_layer_device_cpu"`를 그대로 쓴다(하위 호환). `quant_backend == "nn-linear-fallback"`이면 이 FAIL과 별개로 `reasons`에 `"quant_fallback"`이 항상 추가로 남아 폴백 사실도 함께 드러난다(이 값 자체는 verdict에 영향을 주지 않는 정보성이다).
>
> | `quant_backend` | `device` | 판정 |
> |---|---|---|
> | `bnb-4bit` | `cuda` | PASS |
> | `bnb-4bit` | `cpu` | FAIL (`quant_layer_device_cpu`) |
> | `nn-linear-fallback` | `cuda` | PASS (`reasons`에 정보성 `quant_fallback`만) |
> | `nn-linear-fallback` | `cpu` | FAIL (`quant_layer_device_cpu` + `quant_fallback`) |

> **`memory_delta_mb` WARN은 "예측 대비 이탈"이 아니라 "가용 VRAM 대비 소모율"이다** (2026-08-06 회의 안건 1로 판정 기준 자체가 바뀜). 원래 설계는 raw에 `expected_memory_delta_mb`(probe 기반 외삽, [architecture.md §7](../architecture.md))가 있을 때만 15% 이탈 여부를 평가했는데, MVP에는 그 외삽이 없어 사실상 죽어있는 조건이었다. 대신 canary 기동 직전에 조회한 `env.gpu_free_mb`([`query_gpu_state`](#query_gpu_state-gpupy-w-gpu-산출물) 참고) 대비 `memory_delta_mb`가 90% 이상이면 WARN이다 — 예측 모델 없이도 "이 canary 실행만으로 가용 VRAM을 거의 다 썼다 → 실제 학습(더 큰 배치/시퀀스)에서는 OOM 위험이 크다"는 신호를 바로 줄 수 있다. reason 이름은 `"memory_delta_high"`를 그대로 쓴다(판정 항목 6개 중 하나의 *의미*가 바뀐 것으로 취급 — reason을 새로 늘리지 않는다). `env.gpu_free_mb`가 없으면(NVML 조회 실패 등) 이 WARN은 조용히 건너뛰고 FAIL로 취급하지 않는다.

## `suggest_fix`

판정 결과(`verdict`·`reasons`·`error_log`까지 담긴 딕셔너리)를 받아서 원인을 분류하고 해결 명령어를 만들어 돌려준다. 호출한 쪽은 이 반환값을 리포트에 그대로 찍는다.

```python
def suggest_fix(check_result: dict) -> dict | None:
    """실행하지 않는다 — 명령어만 반환. verdict가 PASS면 None."""
    return {
        "cause": "bnb_not_compiled_with_cuda",
        "message": "bitsandbytes가 CUDA 지원 없이 빌드됨",
        # 표시용(그대로 복사해 쓸 수 있는 문자열)
        "fix_command": "/home/user/venv/bin/python -m pip install bitsandbytes --upgrade --force-reinstall",
        # 실행용(FixExecutor가 그대로 subprocess에 넘긴다)
        "fix_argv": ["/home/user/venv/bin/python", "-m", "pip", "install",
                     "bitsandbytes", "--upgrade", "--force-reinstall"],
    }
```

명령은 PATH의 `pip`이 아니라 `sys.executable`을 앞에 세운다 — 진단한 환경과 다른 환경에 설치되는 것을 막기 위해서다([cli.md](cli.md)의 `fix_command`·`fix_argv` 절). 실행할 명령이 없는 원인은 두 값이 모두 `None`이다. `apply_fix`는 `fix_argv`를 쓰고, 없으면 예전 계약대로 `fix_command`를 `shlex.split`한다.

`reasons`만으로 충분한 경우(WARN 두 개, OOM)는 바로 fix 문구로 이어지고, `import_crash`나 `device=cpu`처럼 원인이 여러 갈래인 경우는 `error_log`나 `env` 같은 부가 정보를 추가로 조회해서 확정한다. **이 부가 정보를 직접 import해서 알아내면 안 된다** — 자식이 `env`에 실어 보낸 값을 읽어야 한다(위 `env` 절 참고). 실제 실행(`--yes`일 때)은 이 함수 밖, FixExecutor에 있다.

**갈래를 가르는 순서는 신호의 신뢰도 순이다.** `fix_command`가 붙는 원인은 곧 `--yes`가 실제로 실행하는 명령이므로, 근거가 약한 신호로 앞질러 확정하면 사용자 환경에 무관한 변경이 실행된다(#51·#55·#72가 전부 이 유형의 버그였다).

| 갈래 | 판정 순서 | cause |
|---|---|---|
| `import_crash` | `error_log`에 `bitsandbytes` 문자열이 있을 때만 bnb 문제로 본다. cuda라는 글자만으로는 아니다(#51) | `bnb_not_compiled_with_cuda` / `import_crash_general` |
| `device=cpu` 폴백 | ① `torch_cuda_version`(CPU 전용 빌드인가) → ② `gpu_free_mb`(NVML이 GPU를 봤는가) → ③ `bnb_compiled_with_cuda`(①②를 못 읽었을 때만) | `torch_cpu_only_build` / `torch_cpu_only_build_no_gpu` / `no_nvidia_gpu_or_driver` / `cuda_device_not_visible` / `bnb_not_compiled_with_cuda` / `4bit_cpu_fallback_other` |

`env`의 속성을 **못 읽은 것**과 그 속성이 **아닌 것**은 다르다 — `torch_version`이 `None`인데 `torch_cuda_version`도 `None`인 것은 "CPU 전용 빌드"가 아니라 "수집 실패"다. 수집 실패를 원인으로 읽으면 멀쩡한 환경에 재설치를 권하게 된다.

**`torch_cpu_only_build`의 `fix_command`는 `env.gpu_driver_version`으로 CUDA 휠 태그를 고른다**(#82). 드라이버 버전 문자열의 major 브랜치 번호만 보는 근사 매핑이다 — 정확한 근거와 표가 어긋나는 조건은 [ADR-0007](../adr/0007-driver-version-based-torch-cuda-wheel-selection.md) 참고. 값이 없거나 매핑에 없으면 기존 기본값(`cu124`)으로 떨어진다.

**`no_nvidia_gpu_or_driver`·`torch_cpu_only_build_no_gpu`·`cuda_device_not_visible`는 `fix_command`가 계속 `None`이다**(#81, #72에서 유보된 판단을 확정). 드라이버 설치는 자동화 등급 D(사람만 실행)이고, GPU가 안 보이는 상태에서 CUDA 빌드 torch를 자동 설치해봐야 아무것도 달라지지 않는다 — 판단 근거는 [ADR-0008](../adr/0008-no-auto-fix-for-user-judgment-causes.md) 참고. `cuda_device_not_visible`만 예외로, 안내 문구에 `env.cuda_visible_devices`의 현재 값을 진단 정보로 덧붙인다(fix_command는 그대로 `None`).

## 전체 흐름

1. `run_canary_check()` 실행 → 원시 측정값
2. `judge_result()`로 `verdict`·`reasons`를 얹는다
3. (verdict가 PASS가 아닐 때만) `suggest_fix()`로 원인 분류 + 해결 명령어 생성
4. 판정 + 해결 명령어를 리포트로 출력한다. **`--yes` 없으면 실행은 여기서 끝난다**
5. (`--yes`가 있을 때만, 그리고 **실행할 명령이 있을 때만**) 명령어를 실제로 실행. 실행할 명령이 없거나 실행이 실패하면 6번을 건너뛰고 그 사실을 `notices`로 알린다
6. (5번이 성공했을 때만) `query_gpu_state()`와 `run_canary_check()`를 다시 호출해 **FIX의 근거가 된 그 체크를 같은 조건으로** 재검증하고, 결과를 그 항목에 갈아끼운다
7. 최종 판정 기준으로 종료코드 반환. `--yes` 없이 4번에서 끝났거나 6번을 건너뛰었다면 2번의 1차 판정이 그대로 쓰인다

`--yes` 없이 실행한 사용자는 화면의 해결 명령어를 직접 실행한 뒤 `preflight check`를 다시 돌려 확인하는 수동 루프를 탄다. `--yes`는 이 과정(5·6번)을 자동화할 뿐이다.

## 관련 문서

- CLI 외부 계약: [cli.md](cli.md)
- 설계 배경: [../architecture.md](../architecture.md)
