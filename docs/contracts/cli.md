# CLI 인터페이스 계약

Preflight 명령어의 입출력 형식. CLI를 감싸는 도구(CI 스크립트 등)가 의존할 수 있는 계약이므로, 변경 시 이 문서를 함께 갱신한다.

## 명령어

| 명령어 | 설명 |
|---|---|
| `preflight check` | 기본 체크 — 모델과 무관하게 GPU/드라이버/CUDA 체인이 살아있는지 확인, 문제 발견 시 수정 명령어 제시 |
| `preflight check --model <name> --batch-size <n> --seq-len <n>` | 목표 모델의 실제 config로 canary를 구성해 목표 크기 그대로 실행, VRAM 실측 (MVP) |
| `preflight check --yes` | 제시된 수정 명령어를 실행하고, 직후 자동으로 재확인까지 수행 |
| `preflight check --json` | 사람이 아닌 다른 도구가 읽을 JSON 형식으로 결과 출력 |

`--model` 지정 시에만 의미를 갖는 학습 설정 세부 옵션(LoRA·양자화·옵티마이저 등)은 MVP에 없다.

**`--batch-size`·`--seq-len`은 모델 체크 전용이다.** `--model` 없이 주면 무시된다 — 기본 체크는 아래 "결과 집계"대로 항상 고정 크기로 돈다.

## 결과 집계

`--model`이 주어지면 canary가 **두 번** 실행되고 결과가 2개가 된다. 그 둘을 하나의 최종 판정으로 접는 규칙이다.

### 실행 순서와 크기

```
① 기본 체크    항상 batch=1, seq=8 고정
② 모델 체크    --model 이 있을 때만. --batch-size / --seq-len 을 그대로 사용
```

**기본 체크의 크기는 사용자가 바꿀 수 없다.** 이 체크의 판정 임계값(CPU 대비 2배 등)이 그 고정 조건에서 뽑은 숫자라([ADR-0004](../adr/0004-canary-size-scaled-by-hidden.md)), 크기가 달라지면 임계값의 근거가 무너진다.

### fail-fast — 기본 체크가 FAIL이면 모델 체크를 건너뛴다

| 기본 체크 | 모델 체크 |
|---|---|
| **FAIL** | **실행하지 않고 생략 사유를 표시한다** |
| WARN | 실행한다 |
| PASS | 실행한다 |

`import_crash`처럼 환경 자체가 깨진 상태면 모델 체크도 **같은 이유로 실패**해 새 정보가 없는데, 목표 크기 실행은 수십 초가 든다. 반면 WARN(예: CPU 대비 배수가 낮음)은 GPU가 살아 있다는 뜻이라 VRAM 실측이 여전히 유효하므로 그대로 진행한다.

### 최종 verdict

```
FAIL 이 하나라도 있으면        →  FAIL
아니고 WARN 이 있으면          →  WARN
전부 PASS                     →  PASS
```

**생략된 항목(`skipped`)은 집계에도 문제 개수에도 넣지 않는다** — 판정된 적이 없기 때문이다. 생략은 기본 체크가 FAIL일 때만 일어나므로 최종 판정은 어차피 FAIL이다.

## 종료 코드

**위에서 집계된 단일 `verdict`** 에 따라 규격화된 종료 코드를 반환한다. `--yes` 옵션으로 수정 후 재확인(`ReVerifier`) 수행 시 1차 판정이 아닌 재확인 최종 결과 기준의 종료 코드를 반환한다.

- `0`: 모든 판정 항목이 **PASS**인 경우
- `1`: 최종 판정이 **FAIL**인 경우
- `2`: FAIL 없이 **WARN**만 포함된 경우

> **주의**: 외부 CI/CD 도구(예: GitHub Actions `run` 스텝, Jenkins 등)는 `0`이 아닌 모든 종료 코드(`1`, `2`)를 파이프라인 실패(Error)로 취급할 수 있다. WARN(`2`)을 파이프라인 중단 오류로 취급하지 않고 후속 작업을 진행하려면 스크립트에서 종료 코드를 명시적으로 분기 처리해야 한다(예: `preflight check || [ $? -eq 2 ]`).

## 출력 예시

```
$ preflight check

✔ Canary 연산 실행              device=cuda · 메모리 이동 확인됨
✔ 실행 시간 12ms                CPU 대비 41배 (정상 범위)
✖ bitsandbytes 4bit 레이어      device=cpu 감지 → 조용한 CPU 폴백

FIX: pip install bitsandbytes --upgrade --force-reinstall
재확인: preflight check --yes

3개 항목 확인 · 1개 문제 발견 · 소요 시간 4초
```

`--model`이 주어지면 기본 체크(GPU/드라이버/CUDA 체인 확인)와 모델 체크(목표 config로
실측)가 순서대로 실행되어 `results`에 2개가 담긴다(기본 체크가 FAIL이면 모델 체크는
생략되지만 `results` 원소는 그대로 2개다 — 위 "결과 집계" 참고. 단일 체크 출력은 위 첫
예시처럼 표제 없이 지금까지와 동일하다 — 결과가 1개뿐일 때만이다). 텍스트 출력은 두 블록을
`기본 체크`/`모델 체크: <model_name>` 표제로 구분해 보여주고, VRAM 실측 줄은 canary가
옮긴 메모리량을 canary 기동 직전 조회한 가용/총 VRAM과 함께 보여준다(`query_gpu_state`,
[canary-api.md](canary-api.md) 참고) — 이 가용 VRAM
숫자가 바로 `judge_result`의 90% 헤드룸 WARN 판정에도 쓰인 숫자와 같다:

```
$ preflight check --model meta-llama/Llama-3.1-8B --batch-size 2 --seq-len 2048

기본 체크
✔ Canary 연산 실행              device=cuda · 메모리 이동 확인됨
✔ 실행 시간 12ms                CPU 대비 41배 (정상 범위)
✔ bitsandbytes 4bit 레이어      device=cuda 정상

모델 체크: meta-llama/Llama-3.1-8B
✔ Canary 연산 실행              device=cuda · 메모리 이동 확인됨
✔ VRAM 실측                     8.4GB / 9.2GB 가용 (총 12GB)
✔ 목표 배치 크기 적합            batch=2, seq=2048 기준

6개 항목 확인 · 문제 없음 · 소요 시간 21초
```

가용 VRAM의 90% 이상을 소모했으면(`memory_delta_high`) VRAM 실측 줄 옆에 `⚠ VRAM 여유`
줄이 추가로 나온다 — WARN 판정과 화면이 항상 같은 숫자를 가리킨다.

실측값이 1GB 미만이면 `18MB`처럼 **MB 단위**로 표기한다 — `0.0GB`로 뭉개지면 측정에 실패한
것처럼 읽히기 때문이다(#45). 가용·총 VRAM은 항상 충분히 커서 GB 그대로다.

실패 항목의 상세 로그(`error_log`)는 화면에서 200자로 줄이되, **트레이스백의 첫 프레임과
마지막 줄(실제 예외 타입·메시지)은 항상 남긴다**(#43) — 앞에서만 자르면 원인 줄이 늘
잘려나가 "에러 로그 확인 필요"라는 안내를 도구 스스로 지키지 못한다. `--json`의 `error_log`는
전문 그대로다. 줄였을 때는 마지막에 `(로그 일부만 표시 — 전문은 preflight check --json)`
한 줄을 덧붙여 잘렸음을 알린다 — 가장 짧은 실패 로그도 200자를 넘어 사실상 모든 에러가 잘리므로,
안내가 없으면 사용자는 잘린 것을 전부라고 믿는다.

기본 체크가 FAIL이면 모델 체크는 실행되지 않고, 그 자리에 **생략 사유 한 줄**만 나온다.
판정 줄이 아니므로 ✔/⚠/✖ 기호를 붙이지 않고 문제 개수에도 넣지 않는다:

```
$ preflight check --model meta-llama/Llama-3.1-8B --batch-size 2 --seq-len 2048

기본 체크
✔ Canary 연산 실행              device=cuda · 메모리 이동 확인됨
✔ 실행 시간 12ms                CPU 대비 41배 (정상 범위)
✖ bitsandbytes 4bit 레이어      device=cpu 감지 → 조용한 CPU 폴백

모델 체크: meta-llama/Llama-3.1-8B
— 환경 체크 실패로 생략

FIX: pip install bitsandbytes --upgrade --force-reinstall
재확인: preflight check --yes

3개 항목 확인 · 1개 문제 발견 · 소요 시간 5초
```

생략 줄은 `total_items`에 들어가지 않아 항목 수가 기본 체크의 3개 그대로다 — 위 첫 예시와 같은 상황이라 줄 구성도 같다.

`--json` 모드는 이 표제와 무관하게 `results` 배열에 두 판정 결과를 순서 그대로(기본
체크가 index 0) 가공 없이 담는다 — 표제는 텍스트 출력 전용이다.

`--json`은 사람이 읽는 위 출력 대신, 다른 도구가 파싱하기 쉬운 아래 구조를 표준출력에 그대로 찍는다(`print()`로 출력 — 파이핑 시 색상 코드 등이 섞이지 않는다). `results`는 `judge_result()` 출력(및 있다면 `suggest_fix()` 결과를 병합한 `"fix"` 키)을 가공 없이 그대로 담고, `summary`는 위 요약 줄과 동일한 집계다. `exit_code_hint`는 WARN·FAIL 유무를 미리 계산해둔 참고값일 뿐 — 실제 종료 코드는 CLI 진입점이 결정한다.

> `summary.total_items`는 `results` 배열의 원소 개수가 아니라, 텍스트 모드에서 찍히는 항목 줄 개수(위 기본 체크 예시의 "3개 항목")와 같은 값이다 — 아래 예시는 canary 실행 1건(`results` 원소 1개)에서 status·실행시간·quant 세 줄이 나오는 경우라 `total_items`가 3이다.

```
$ preflight check --json

{
  "results": [
    {
      "status": "ok",
      "device": "cpu",
      "memory_delta_mb": 130.7,
      "elapsed_ms": 1.8,
      "cpu_multiplier": 19.0,
      "quant_backend": "bnb-4bit",
      "error_log": null,
      "verdict": "FAIL",
      "reasons": ["quant_layer_device_cpu"],
      "fix": {
        "cause": "bnb_not_compiled_with_cuda",
        "message": "bitsandbytes가 CUDA 지원 없이 빌드됨",
        "fix_command": "pip install bitsandbytes --upgrade --force-reinstall"
      }
    }
  ],
  "summary": { "total_items": 3, "pass": 2, "warn": 0, "fail": 1, "elapsed_seconds": 4.0 },
  "exit_code_hint": 1
}
```

## 리포트 입력 스키마

`render_report()`에 넘기는 각 항목은 `judge_result()` 결과에 **CLI 진입점이 자기가 아는 값을 얹은 것**이다. 아래 필드는 `run_canary_check()`의 반환 스키마([canary-api.md](canary-api.md))에 없다 — 자식이 **잰** 값이 아니라 CLI가 명령줄에서 **받은** 값이거나 CLI가 판단한 값이라, 자식에게 보냈다가 되돌려받을 이유가 없기 때문이다.

| 필드 | 언제 | 무엇 |
|---|---|---|
| `model_name` | 모델 체크 항목에만 | `--model` 값. 표제(`모델 체크: <name>`)와 **기본/모델 체크 구분**에 쓰인다 |
| `batch_size` · `seq_len` | 모델 체크 항목에만 | `--batch-size`·`--seq-len` 값. "목표 배치 크기 적합" 줄에 쓰인다 |
| `skipped` | 생략된 항목에만 | 생략 사유 문자열. 이 키가 있으면 `verdict`·`reasons`가 없다 |
| `fix` | `verdict != "PASS"`인 항목에만 | `suggest_fix()` 반환값 |

**기본 체크와 모델 체크는 `model_name`의 유무로 구분한다.** 순서(index)나 `cpu_multiplier`가 `None`인지로 추론하지 않는다 — 기본 체크도 CPU 기준선 측정에 실패하면 `cpu_multiplier`가 `None`이라 오인된다.

생략된 항목은 판정을 받은 적이 없으므로 판정 필드가 통째로 없다.

```python
[
    {..., "verdict": "FAIL", "reasons": ["quant_layer_device_cpu"], "fix": {...}},
    {"model_name": "meta-llama/Llama-3.1-8B", "batch_size": 2, "seq_len": 2048,
     "skipped": "환경 체크 실패"},
]
```

리포트는 이 항목에 대해 판정 줄 대신 **생략 사유 한 줄**을 그리고, 문제 개수(`summary.fail`·`total_items`)에 넣지 않는다.

## 관련 문서

- 내부 모듈 간 API 계약: [canary-api.md](canary-api.md)
- 설계 배경: [../architecture.md](../architecture.md)
