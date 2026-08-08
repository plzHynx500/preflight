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
    "bnb_compiled_with_cuda": True,
    "bnb_cpu_4bit_supported": True,     # 4bit을 실제로 시도한 경우에만. 아니면 None
}
```

**자식이 읽어야 한다.** 이 값들을 읽으려면 `torch`·`bitsandbytes`를 import해야 하는데, 진단 대상이 바로 *"그 import가 죽는 환경"* 이다. 부모가 읽으면 원인을 확인하려다 CLI까지 함께 죽어 FR-03 격리가 무너진다([ADR-0002](../adr/0002-subprocess-isolation-for-canary.md), Issue #19).

항목은 **독립적으로 실패**할 수 있고, 실패한 항목만 `None`이 된다. `status == "import_crash"`여서 아무 속성도 못 읽은 경우에도 **키는 유지한 채 값만 전부 `None`인 dict**가 온다 — 소비자가 키 존재 여부까지 따로 방어하지 않게 하기 위함이다. 이때 속성을 다시 읽으려 시도하지는 않는다(방금 실패한 import를 반복할 뿐이다).

`env` 자체를 못 받는 경우(구버전 canary 등)는 `None`이므로, 소비자는 `(raw.get("env") or {}).get(...)` 형태로 읽어야 한다.

> **`env`는 여럿이 쌓아가는 칸이다 — 통째로 갈아끼우지 말 것.** 자식이 라이브러리 상태를 채운 뒤, 부모(`cli`)가 GPU 정보를 **얹는다**. `raw["env"] = {...}`로 대입하면 자식이 채운 값이 사라지고, **에러 없이 원인 분류만 조용히 실패**해 잡기 어렵다.
>
> ```python
> raw["env"] = {**(raw.get("env") or {}), "gpu_free_mb": ..., "gpu_total_mb": ...}
> ```
>
> **`setdefault("env", {})`를 쓰면 안 된다.** `_normalize()`가 모든 필드를 항상 만들기 때문에 `env`는 **키가 있고 값만 `None`인 상태**가 되는데, `setdefault`는 키 존재 여부만 보므로 `None`을 그대로 돌려준다 → `None.update(...)`로 **부모 프로세스가 죽는다.** 자식이 아예 안 돌아 부모가 결과를 직접 만드는 경로(기동 실패·타임아웃·결과 파일 없음)에서도 `env`는 `None`이므로, 병합 코드는 어떤 경우에도 `None`을 견뎌야 한다.

### `rss_peak_mb` — 그 시점까지의 호스트 RAM 최고점

`status == "error"`의 원인 후보를 좁히기 위한 **내부 근거**다. `verdict`에 영향을 주지 않고 화면에도 찍지 않는다 — 다만 **원인 분류(`suggest_fix`)에는 쓸 수 있다.** 이유는 [ADR-0006](../adr/0006-ram-recorded-internally-only.md) 참고.

자식은 [사전 기록](../adr/0005-pre-written-result-over-exit-code.md)을 남길 때마다 이 값을 다시 잰다. 최종 결과에만 넣으면 정작 예외 없이 즉사했을 때 아무것도 남지 않는다.

**현재값이 아니라 최고점**(Windows `PeakWorkingSetSize` · Linux `VmHWM`)이다. 이 값을 읽는 시점은 언제나 사후라 *"지금 얼마나 쓰고 있나"* 는 의미가 없고, 커널이 최고점을 대신 기록해주므로 우리가 주기적으로 샘플링하지 않아도 torch import처럼 잠깐 치솟았다 내려가는 구간이 그대로 남는다.

> **한계 — 실행 단계 도중의 증가는 못 잡는다.** 사전 기록은 각 단계에 **진입하기 전에** 쓰이므로, 실행 중 RAM이 급증한 뒤 SIGKILL로 죽으면 살아남는 값은 그 직전 단계까지의 최고점이다. 죽기 직전까지 추적하려면 부모가 자식을 폴링해야 하는데(`subprocess.run` → `Popen` + 루프) MVP 범위 밖이다 — **그때도 이 필드의 이름·의미는 그대로고 누가 재느냐만 바뀐다**(#27).
>
> 같은 이유로 **`suggest_fix`가 아직 이 값을 읽지 않는다.** *"다른 앱이 RAM을 다 써서 우리가 희생된"* 경우를 가르려면 시스템 여유 RAM이 함께 있어야 하는데, 그것도 #27에 함께 남겼다.

측정 실패 시 `0`이 아니라 `None`이다 — `0`은 *"RAM을 안 썼다"* 는 뜻이 되어 정반대 결론으로 이끈다. macOS는 두 경로가 모두 없어 항상 `None`이다(NFR-01 대상 아님).

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
| FAIL | `status == "oom"` · `status == "import_crash"` · 4bit 레이어 device=cpu 감지 |
| WARN | `memory_delta_mb`가 예측 대비 15% 이상 벗어남 · `cpu_multiplier < 2` |
| PASS | 위 조건에 전부 해당하지 않음 |

FAIL 3개·WARN 2개, 총 5개 판정 항목 전부 MVP 구현으로 팀이 확정했다. 두 숫자(15%, 2배)는 정밀 검증된 값이 아니라 매직넘버로 우선 채택한 것이며, 실측 데이터가 쌓이면 조정한다.

> "4bit 레이어 device=cpu" 조건은 `quant_backend == "bnb-4bit"`일 때만 판정할 수 있다 — `"nn-linear-fallback"`이면 애초에 4bit 레이어가 없으므로 이 항목은 판정 대상에서 빠지고, 대신 폴백이 발동됐다는 사실이 리포트에 드러나야 한다.

## `suggest_fix`

판정 결과(`verdict`·`reasons`·`error_log`까지 담긴 딕셔너리)를 받아서 원인을 분류하고 해결 명령어를 만들어 돌려준다. 호출한 쪽은 이 반환값을 리포트에 그대로 찍는다.

```python
def suggest_fix(check_result: dict) -> dict | None:
    """실행하지 않는다 — 명령어 텍스트만 반환. verdict가 PASS면 None."""
    return {
        "cause": "bnb_not_compiled_with_cuda",
        "message": "bitsandbytes가 CUDA 지원 없이 빌드됨",
        "fix_command": "pip install bitsandbytes --upgrade --force-reinstall",
    }
```

`reasons`만으로 충분한 경우(WARN 두 개, OOM)는 바로 fix 문구로 이어지고, `import_crash`나 `device=cpu`처럼 원인이 여러 갈래인 경우는 `error_log`나 `env.bnb_compiled_with_cuda` 같은 부가 정보를 추가로 조회해서 확정한다. **이 부가 정보를 직접 import해서 알아내면 안 된다** — 자식이 `env`에 실어 보낸 값을 읽어야 한다(위 `env` 절 참고). 실제 실행(`--yes`일 때)은 이 함수 밖, FixExecutor에 있다.

## 전체 흐름

1. `run_canary_check()` 실행 → 원시 측정값
2. `judge_result()`로 `verdict`·`reasons`를 얹는다
3. (verdict가 PASS가 아닐 때만) `suggest_fix()`로 원인 분류 + 해결 명령어 생성
4. 판정 + 해결 명령어를 리포트로 출력한다. **`--yes` 없으면 실행은 여기서 끝난다**
5. (`--yes`가 있을 때만) 명령어를 실제로 실행
6. (`--yes`가 있을 때만) `run_canary_check()`를 다시 호출해 재검증
7. 최종 판정 기준으로 종료코드 반환. `--yes` 없이 4번에서 끝났다면 2번의 1차 판정이 그대로 쓰인다

`--yes` 없이 실행한 사용자는 화면의 해결 명령어를 직접 실행한 뒤 `preflight check`를 다시 돌려 확인하는 수동 루프를 탄다. `--yes`는 이 과정(5·6번)을 자동화할 뿐이다.

## 관련 문서

- CLI 외부 계약: [cli.md](cli.md)
- 설계 배경: [../architecture.md](../architecture.md)
