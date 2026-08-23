# CLI 인터페이스 계약

Preflight 명령어의 입출력 형식. CLI를 감싸는 도구(CI 스크립트 등)가 의존할 수 있는 계약이므로, 변경 시 이 문서를 함께 갱신한다.

## 명령어

| 명령어 | 설명 |
|---|---|
| `preflight check` | 기본 체크 — 모델과 무관하게 GPU/드라이버/CUDA 체인이 살아있는지 확인, 문제 발견 시 수정 명령어 제시 |
| `preflight check --model <name> --batch-size <n> --seq-len <n>` | 목표 모델의 실제 config로 canary를 구성해 목표 크기 그대로 실행, VRAM 실측 (MVP) |
| `preflight check --yes` | 제시된 수정 명령어를 실행하고, 직후 자동으로 재확인까지 수행 |
| `preflight check --json` | 사람이 아닌 다른 도구가 읽을 JSON 형식으로 결과 출력 |
| `preflight --version` | 설치된 preflight 버전(`preflight.__version__`)을 출력하고 종료 |

`--model` 지정 시에만 의미를 갖는 학습 설정 세부 옵션(LoRA·양자화·옵티마이저 등)은 MVP에 없다.

**`--batch-size`·`--seq-len`은 모델 체크 전용이다.** `--model` 없이 주면 무시된다 — 기본 체크는 아래 "결과 집계"대로 항상 고정 크기로 돈다.

**`--batch-size`·`--seq-len`은 1 이상의 정수만 허용한다.** 0·음수를 주면 Typer가 파싱 단계에서 즉시 거부한다(`Invalid value for '--batch-size': 0 is not in the range x>=1.`, 종료 코드 2) — canary를 실행하기 전이라 결과 항목이 아예 생기지 않는다(#59).

**`--model`에 빈 문자열(`""`)을 주면 Typer가 파싱 단계에서 즉시 거부한다**(`Invalid value for '--model': ...`, 종료 코드 2). 빈 문자열은 파이썬에서 falsy라 미지정(`None`)과 구분 없이 처리되면 모델 체크 자체가 조용히 사라진다 — CI에서 `--model "$MODEL"`의 변수가 비어 있을 때 특히 위험하다(#126). `--model`을 아예 안 주면(`None`) 지금까지와 동일하게 기본 체크만 실행된다. 공백만 있는 문자열(`"   "`)은 이 검증 대상이 아니다 — "주어진 값"으로 그대로 모델 체크가 시도된다.

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

## `--yes` — 수정 실행과 재확인

`--yes`는 화면에 제시된 수정 명령을 실제로 실행하고 재확인까지 자동화한다. 세 갈래로 갈린다.

| 상황 | 동작 |
|---|---|
| 실행할 명령이 있다 | 명령 실행 → **그 FIX의 근거가 된 체크**를 같은 조건으로 재실행 → 결과 갈아끼움 |
| 실행할 명령이 없다(`fix_command`가 `null`) | 아무것도 실행하지 않고 `notices`로 알린다. **canary를 다시 돌리지 않는다** |
| 명령 실행이 실패했다 | 재확인을 건너뛰고 `notices`로 알린다. 트레이스백은 노출하지 않는다 |

**재확인 대상은 FIX가 붙은 그 항목 하나다.** 그 항목의 `model_name`·`batch_size`·`seq_len`을 그대로 물려 재실행한다 — 기본 체크가 FAIL이라 모델 체크가 생략된 상태에서 `--model`이 주어졌다는 이유로 한 번도 실행된 적 없는 모델 canary를 돌리지 않는다. 재확인 경로도 canary 기동 직전에 `query_gpu_state()`를 **다시** 조회해 `env`에 병합한다(값이 1차와 달라지는 것이 맞는 동작이다 — 알고 싶은 건 수정 이후 시점의 여유 VRAM이고, 이 값이 없으면 `memory_delta_high` WARN이 구조적으로 다시 나올 수 없다).

재확인한 항목은 `results`에서 **재확인 결과로 교체된다**. 이때 CLI가 얹은 값(`model_name`·`batch_size`·`seq_len`)과 실행한 `fix`는 유지되고 `reverified: true`가 붙는다. 재확인하지 않은 나머지 항목은 1차 판정 그대로 남는다.

**기본 체크 재확인이 FAIL을 벗어나면, 생략됐던 모델 체크를 이어서 실행한다(#84).** 1차 fail-fast 규칙(위 "결과 집계")을 재확인에도 그대로 적용한 것이다 — 그러지 않으면 모델을 한 번도 확인한 적이 없는데 최종 판정에는 그 항목이 통째로 빠져 있어, 기본 체크만 PASS면 종료 코드가 0이 되어버린다. `--model`로 물은 "이 모델이 이 기계에서 도는가"라는 질문에 실제로 답한 적이 없는데 성공 취급되는 것이다. 이때도 모델 canary 기동 직전에 `query_gpu_state()`를 다시 조회한다(기본 체크 재확인과 같은 이유). 새로 실행한 모델 체크가 FAIL/WARN이어도 **두 번째 FIX 블록은 나오지 않는다** — FIX는 MVP에서 여전히 하나뿐이다. `notices`에는 "--yes: 기본 체크 통과로 생략됐던 모델 체크(\<model_name\>)를 이어서 실행했다" 줄이 추가된다.

기본 체크 재확인이 여전히 FAIL이면 모델 체크는 실행하지 않는다 — 생략 사유 그대로 `results`에 남고, 불필요한 수십 초짜리 canary 실행을 피한다.

### FIX를 붙일 항목 선택

`verdict`가 PASS가 아닌 항목이 여럿이면 **FAIL을 WARN보다 먼저** 본다. FAIL이 여럿이면 그중 첫 번째다. 배열 순서만 보면 기본 체크 WARN + 모델 체크 FAIL에서 WARN에 FIX가 붙어, 화면은 FAIL을 띄워놓고 안내는 WARN 것을 하게 된다. FIX 블록은 MVP에서 **하나만** 나온다.

## 종료 코드

**위에서 집계된 단일 `verdict`** 에 따라 규격화된 종료 코드를 반환한다.

- `0`: 모든 판정 항목이 **PASS**인 경우
- `1`: 최종 판정이 **FAIL**인 경우
- `2`: FAIL 없이 **WARN**만 포함된 경우

`--yes`로 재확인을 수행했으면, 재확인 결과로 교체된 `results` **전체를 다시 집계한** 값이 기준이다 — 재확인 1건의 판정으로 전체를 대체하지 않는다(그러면 재확인하지 않은 항목의 WARN·FAIL이 종료 코드에서 사라진다). 재확인이 일어나지 않았으면(실행할 명령이 없었거나 명령 실행이 실패했으면) 1차 판정이 그대로 기준이다 — 수정 실패에 별도 종료 코드를 두지 않는다.

> **주의**: 외부 CI/CD 도구(예: GitHub Actions `run` 스텝, Jenkins 등)는 `0`이 아닌 모든 종료 코드(`1`, `2`)를 파이프라인 실패(Error)로 취급할 수 있다. WARN(`2`)을 파이프라인 중단 오류로 취급하지 않고 후속 작업을 진행하려면 스크립트에서 종료 코드를 명시적으로 분기 처리해야 한다(예: `preflight check || [ $? -eq 2 ]`).

## 출력 예시

```
$ preflight check

✔ Canary 연산 실행              device=cuda · 메모리 이동 확인됨
✔ 실행 시간 12ms                CPU 대비 41배 (정상 범위)
✖ bitsandbytes 4bit 레이어      device=cpu 감지 → 조용한 CPU 폴백

FIX: /home/user/venv/bin/python -m pip install bitsandbytes --upgrade --force-reinstall
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

4bit 구성에 실패해 fp32로 폴백한 경우(`quant_backend="nn-linear-fallback"`)에는 모델 체크
블록에도 폴백 한 줄이 나온다. 기본 체크의 `bitsandbytes 4bit 레이어` 줄과 달리 **판정이
아니라 바로 위 VRAM 숫자의 전제를 밝히는 정보성 줄**이라 ✔/⚠/✖ 대신 `ℹ`를 쓰고 문제 개수에
넣지 않는다 — 폴백은 QLoRA가 아니라 fp32 전체 모델을 돌리므로 같은 모델·같은 배치라도
실측값이 크게 나오고, 이 줄이 없으면 사용자는 그 숫자를 QLoRA 기준으로 읽어 "이 GPU로는
무리"라는 정반대 결론을 낸다(#66):

```
모델 체크: meta-llama/Llama-3.1-8B
✔ Canary 연산 실행              device=cuda · 메모리 이동 확인됨
✔ VRAM 실측                     8.4GB / 9.2GB 가용 (총 12GB)
ℹ 4bit 레이어 폴백              4bit 레이어 구성 실패 → fp32 전체 모델로 실측됨 (VRAM 수치가 QLoRA 기준보다 크다)
```

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

FIX: /home/user/venv/bin/python -m pip install bitsandbytes --upgrade --force-reinstall
재확인: preflight check --yes

3개 항목 확인 · 1개 문제 발견 · 소요 시간 5초
```

생략 줄은 `total_items`에 들어가지 않아 항목 수가 기본 체크의 3개 그대로다 — 위 첫 예시와 같은 상황이라 줄 구성도 같다.

`--json` 모드는 이 표제와 무관하게 `results` 배열에 두 판정 결과를 순서 그대로(기본
체크가 index 0) 가공 없이 담는다 — 표제는 텍스트 출력 전용이다.

`--json`은 사람이 읽는 위 출력 대신, 다른 도구가 파싱하기 쉬운 아래 구조를 표준출력에 그대로 찍는다(`print()`로 출력 — 파이핑 시 색상 코드 등이 섞이지 않는다). `results`는 `judge_result()` 출력(및 있다면 `suggest_fix()` 결과를 병합한 `"fix"` 키)을 가공 없이 그대로 담고, `summary`는 위 요약 줄과 동일한 집계다. `exit_code_hint`는 위 "종료 코드" 절의 규칙(0/1/2)을 미리 계산해둔 참고값이다 — **실제 종료 코드와 항상 같은 값**이며, 실제 종료 코드는 CLI 진입점이 결정한다(#70 전에는 0/1 이진 계산이라 WARN도 FAIL과 같은 1이 나가는 버그가 있었다).

`notices`는 **판정이 아니라 도구가 한 일**을 담는 문자열 배열이다(`--yes`가 실행한 명령, 실행할 명령이 없었다는 사실, 수정 실패). 텍스트 모드에서는 요약 줄 앞에 같은 내용이 나온다. `--yes` 없이 실행하면 항상 빈 배열이다. 수정 명령이 실패해도 pip의 stdout/stderr는 싣지 않는다 — 그대로 흘리면 화면이 길게 쏟아지므로, 대신 명령을 직접 실행해 확인하라고 안내한다.

## 진행 표시 (stderr)

기본 체크만도 첫 줄까지 10초 안팎(torch import + canary 실행), `--yes`는 pip 재설치 + canary
재실행으로 1분 넘게 아무 출력이 없어 멈춘 것처럼 보인다(#63). `check`는 오래 걸리는 단계
직전에 아래 안내를 **stderr**로 한 줄씩 찍는다 — `results`/`summary`/`notices`가 담기는
stdout(`--json`이 아니어도 텍스트 리포트도 stdout)과는 별개다.

| 시점 | 문구 |
|---|---|
| canary 실행 직전 (기본 체크·모델 체크·`--yes`로 재개된 모델 체크 모두) | `진단 중… (torch 불러오기 · canary 실행, 수 초~수십 초)` |
| `--yes`가 수정 명령을 실행하기 직전 | `수정 명령 실행 중: <fix_command>` |
| `--yes`가 재확인(reverify)을 시작하기 직전 | `재확인 중…` |

실행할 수정 명령이 없는 경우(`fix_argv`가 `null`)는 `apply_fix`/`reverify`를 아예 부르지
않으므로 뒤 두 줄도 찍히지 않는다.

`--json`은 이 절과 무관하다 — stdout에는 여전히 `print(json.dumps(...))` 한 번뿐이라
`preflight check --json > result.json`처럼 리다이렉트해도 stderr의 진행 표시가 섞여
들어가지 않는다.

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
      "env": {
        "torch_version": "2.13.0+cpu",
        "torch_cuda_version": null,
        "bnb_compiled_with_cuda": null,
        "bnb_cpu_4bit_supported": null,
        "gpu_free_mb": 9420.0,
        "gpu_total_mb": 12282.0,
        "gpu_driver_version": "560.94.03",
        "gpu_name": "NVIDIA GeForce RTX 4070 Ti"
      },
      "fix": {
        "cause": "torch_cpu_only_build",
        "message": "설치된 torch가 CPU 전용 빌드다 (torch.version.cuda 없음) — NVIDIA GPU는 감지됐으므로 CUDA 빌드 torch로 재설치하면 GPU를 쓸 수 있다",
        "fix_command": "/home/user/venv/bin/python -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124",
        "fix_argv": ["/home/user/venv/bin/python", "-m", "pip", "install", "--force-reinstall", "torch", "--index-url", "https://download.pytorch.org/whl/cu124"]
      }
    }
  ],
  "summary": { "total_items": 3, "pass": 2, "warn": 0, "fail": 1, "elapsed_seconds": 4.0 },
  "notices": [],
  "exit_code_hint": 1
}
```

`fix_command`는 **화면 표시용이자 사용자가 그대로 복사해 쓸 문자열**이고, `fix_argv`는 **실제로 실행되는 인자 리스트**다. 둘 다 PATH의 `pip`이 아니라 `sys.executable`(= 지금 preflight를 돌리고 있는 파이썬)을 가리킨다 — venv를 활성화하지 않았거나 pipx·전역 설치로 쓰는 환경에서 "진단은 A 환경, 수정은 B 환경"이 되면 재확인이 계속 실패해도 사용자는 이유를 알 수 없다. 짧게 보이려고 `python -m pip`으로 줄이지 않는 이유도 같다 — 복사해 붙이는 순간 다시 "지금 활성화된 파이썬"으로 돌아간다. 공백이 든 경로(`C:\Program Files\...`)는 `fix_command`에서 큰따옴표로 감싸고, 실행은 `fix_argv`를 쓰므로 그 문자열을 다시 파싱하지 않는다. 실행할 명령이 없는 원인은 두 값이 모두 `null`이다.

## 리포트 입력 스키마

`render_report()`에 넘기는 각 항목은 `judge_result()` 결과에 **CLI 진입점이 자기가 아는 값을 얹은 것**이다. 아래 필드는 `run_canary_check()`의 반환 스키마([canary-api.md](canary-api.md))에 없다 — 자식이 **잰** 값이 아니라 CLI가 명령줄에서 **받은** 값이거나 CLI가 판단한 값이라, 자식에게 보냈다가 되돌려받을 이유가 없기 때문이다.

| 필드 | 언제 | 무엇 |
|---|---|---|
| `model_name` | 모델 체크 항목에만 | `--model` 값. 표제(`모델 체크: <name>`)와 **기본/모델 체크 구분**에 쓰인다 |
| `batch_size` · `seq_len` | 모델 체크 항목에만 | `--batch-size`·`--seq-len` 값. "목표 배치 크기 적합" 줄에 쓰인다 |
| `skipped` | 생략된 항목에만 | 생략 사유 문자열. 이 키가 있으면 `verdict`·`reasons`가 없다 |
| `fix` | FIX 대상으로 뽑힌 항목에만 | `suggest_fix()` 반환값. `--yes`로 그 수정을 실행한 뒤에는 재확인 결과가 PASS여도 **어떤 수정을 했는지 남기려고 그대로 유지된다** |
| `reverified` | `--yes`로 재확인한 항목에만 | `True`. 표제에 `(재확인)`을 붙이는 데 쓰인다 |

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
