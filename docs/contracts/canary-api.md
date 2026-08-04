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
    }
```

**첫 인자는 모델 객체가 아니라 모델명이다.** 부모가 `nn.Module`을 만들어 넘기려면 부모 프로세스가 `torch`를 import해야 하는데, `import torch`·`import bitsandbytes` 자체가 죽는 상황을 잡는 것이 프로세스 격리([ADR-0002](../adr/0002-subprocess-isolation-for-canary.md))의 목적이므로 그 순간 격리가 무력화된다. 따라서 모델 구성은 전부 자식 프로세스 안에서 일어난다 — 모델을 만드는 코드(`canary/model.py`)도 자식에서만 호출되므로 함수 안에서만 torch/transformers를 import해야 하고, 부모 쪽 모듈에 import를 노출하면 안 된다. `model_name`이 `None`이면 기본 체크(모델과 무관한 최소 대표 구조)를 뜻한다.

`status`는 사람이 읽는 값이 아니라 다음 단계가 기계적으로 분기하는 스위치다. `"import_crash"`·`"oom"`은 각각 재설치·batch 축소라는 서로 다른 FIX로 이어지므로, 그 둘로 확정할 수 없는 실패(모델명 오타로 config 조회 실패, 미구현 경로 호출 등)를 둘 중 하나로 적으면 사용자에게 엉뚱한 해결 명령이 제시된다. 그런 실패는 `"error"`로 두고 `error_log`에 원본 메시지를 담는다.

`quant_backend`는 canary가 실제로 어떤 연산 경로로 돌았는지를 나타낸다. 구버전 bitsandbytes 등으로 4bit 레이어 구성·실행이 실패해 평범한 `nn.Linear`로 대체된 경우 `"nn-linear-fallback"`이 되며([architecture.md §6-01](../architecture.md)), 이때는 "4bit 레이어 device=cpu" 판정을 할 수 없으므로 리포트에 폴백 사실을 명시해야 한다.

| 필드 | 채우는 쪽 | 쓰는 쪽 |
|---|---|---|
| `model_name`, `batch_size`, `seq_len` (입력) | CLI 진입점 — 모델명(또는 기본 체크면 `None`)만 넘긴다 | 실행·측정 담당(자식 프로세스에서 모델 구성) |
| `status`/`device`/`memory_delta_mb`/`elapsed_ms`/`cpu_multiplier`/`quant_backend`/`error_log` (출력) | 실행·측정·크래시 캐치까지 전부 담당 | 판정(`judge_result`) → 원인분류(`suggest_fix`) |

## `build_dummy_model` / `build_dummy_input` (canary/model.py, W4 산출물)

`run_canary_check`의 자식 프로세스 안에서만 호출되는 내부 헬퍼다 — 모듈 경계 밖으로
노출되지 않지만, W9 통합(worker.py의 `--model` 경로 연결) 때 그대로 쓰일 시그니처라
여기 함께 적는다.

```python
def build_dummy_model(model_name: str):
    """가중치 다운로드 없이 config만 조회해 랜덤 초기화 모델을 만든다."""
    ...
    return model, config  # config.vocab_size를 build_dummy_input에 넘기기 위해 함께 반환

def build_dummy_input(batch_size: int, seq_len: int, vocab_size: int):
    """토큰 ID(정수) 텐서. vocab_size가 있어야 유효한 범위의 ID를 만들 수 있다."""
    ...
```

`build_dummy_input`은 원래 `(batch_size, seq_len)`만 받는 시그니처로 스텁이 있었으나,
`--model` 경로의 입력은 정수 토큰 ID라 `vocab_size`(=`build_dummy_model`이 돌려준
`config.vocab_size`)가 있어야 만들 수 있어 W4 구현 중 `vocab_size` 파라미터를
추가했다. W9에서 두 함수를 이어 쓸 때는 `build_dummy_model`이 돌려준 `config`에서
`vocab_size`를 꺼내 `build_dummy_input`에 그대로 넘기면 된다.

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
| WARN | `memory_delta_mb`가 예측 대비 15% 이상 벗어남 · `cpu_multiplier < 2` |
| PASS | 위 조건에 전부 해당하지 않음 |

FAIL 4개·WARN 2개, 총 6개 판정 항목 전부 MVP 구현으로 팀이 확정했다(`status == "error"`는 W5 구현 중 추가). 두 숫자(15%, 2배)는 정밀 검증된 값이 아니라 매직넘버로 우선 채택한 것이며, 실측 데이터가 쌓이면 조정한다.

> **`device == "cpu"` FAIL은 `quant_backend`와 무관하게 적용된다** (2026-08-03, #18로 발견된 계약 구멍 수정). 기본 체크의 목적 자체가 "GPU/드라이버/CUDA 체인이 물리적으로 살아있는가"([architecture.md §3](../architecture.md))라서, 4bit 레이어 유무와 무관하게 device가 cpu면 그 자체로 실패다. 원래는 `quant_backend == "bnb-4bit"`일 때만 이 조건을 봤는데, 그러면 bitsandbytes 자체가 없어 4bit 레이어가 없는 환경(`quant_backend == "nn-linear-fallback"`)은 이 규칙이 발동하지 못해 GPU가 전혀 없는데도 PASS가 나가는 구멍이 있었다. reason 이름은 `"quant_layer_device_cpu"`를 그대로 쓴다(하위 호환). `quant_backend == "nn-linear-fallback"`이면 이 FAIL과 별개로 `reasons`에 `"quant_fallback"`이 항상 추가로 남아 폴백 사실도 함께 드러난다(이 값 자체는 verdict에 영향을 주지 않는 정보성이다).
>
> | `quant_backend` | `device` | 판정 |
> |---|---|---|
> | `bnb-4bit` | `cuda` | PASS |
> | `bnb-4bit` | `cpu` | FAIL (`quant_layer_device_cpu`) |
> | `nn-linear-fallback` | `cuda` | PASS (`reasons`에 정보성 `quant_fallback`만) |
> | `nn-linear-fallback` | `cpu` | FAIL (`quant_layer_device_cpu` + `quant_fallback`) |

> **`memory_delta_mb` 예측 비교는 현재 사실상 대기 상태다.** 15% 이탈 판정은 raw에 `expected_memory_delta_mb`(옵션, 위 7개 필드 스키마에는 없음)가 있을 때만 평가된다. 이 예측값은 probe 기반 외삽([architecture.md §7](../architecture.md))이 있어야 생기는데 MVP에는 아직 없어서, 이 필드가 없는 한 이 WARN 조건은 발동하지 않는다 — 팀 미결 사항(Notion "!내부용! 논의사항" §WARN 트리거 조건 참고).

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

`reasons`만으로 충분한 경우(WARN 두 개, OOM)는 바로 fix 문구로 이어지고, `import_crash`나 `device=cpu`처럼 원인이 여러 갈래인 경우는 `error_log`나 `compiled_with_cuda` 같은 부가 정보를 추가로 조회해서 확정한다. 실제 실행(`--yes`일 때)은 이 함수 밖, FixExecutor에 있다.

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
