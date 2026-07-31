# Preflight — 아키텍처

이 문서는 Preflight의 시스템 구조와 모듈 설계, 구현 시 주의사항을 다룬다. 제품 기획(문제 정의, MVP 범위, 일정, 팀 구성 등)은 Notion 원본이 기준이다 — 이 문서는 "무엇을/왜 만드는지"가 아니라 "어떻게 만드는지"만 다룬다. `AGENTS.md`에 정리된 문서 소유권 규칙을 따른다.

## 목차
1. [시스템 개요](#1-시스템-개요)
2. [기술 스택](#2-기술-스택)
3. [Canary 검증 구성 (MVP)](#3-canary-검증-구성-mvp)
4. [공통 처리 파이프라인](#4-공통-처리-파이프라인)
5. [모듈 구성](#5-모듈-구성)
6. [구현 시 주의사항](#6-구현-시-주의사항)
7. [향후 확장 설계 — probe 기반 외삽 + fail-fast](#7-향후-확장-설계--probe-기반-외삽--fail-fast)

---

## 1. 시스템 개요

Preflight는 환경 설치 단계와 학습 실행 단계 사이에 놓이는 검증 게이트다. 드라이버·CUDA 같은 하위 레이어나 다른 학습 도구를 대체하지 않고, 정적 진단과 달리 실제 연산 실행(canary)을 중심에 둔다.

```
기존 환경             CanaryEngine          FixExecutor              ReVerifier          출력
(드라이버·CUDA·   →   (subprocess 격리  →   (기본 dry-run       →   (CanaryEngine   →   (터미널 /
 PyTorch·라이브러리)     실행)                --yes 시 실행)           재실행)              JSON / 종료코드)
```

실측 위주 설계인 이유, subprocess 격리를 채택한 이유, 상대 비교 타이밍을 쓰는 이유 등 장기적으로 다시 물어볼 법한 결정은 `adr/`에 별도로 남겨둔다.

---

## 2. 기술 스택

| 구분 | 선택 | 이유 |
|---|---|---|
| 언어 | Python 3.9+ | 사용자의 실제 PyTorch/CUDA 환경을 그 자리에서 불러와 검사해야 하므로 사실상 유일한 선택지 |
| CLI 프레임워크 | Typer | 타입 힌트 기반으로 명령어를 간결하게 정의, `--help` 자동 생성 |
| 출력 포맷팅 | rich | 터미널에 색상·기호(✔/⚠/✖)를 안정적으로 렌더링 |
| GPU 상태 조회 | pynvml | nvidia-smi가 쓰는 NVML C 라이브러리의 Python 바인딩 |
| 학습 스택 연동 | 동일 인터프리터 내 import | 사용자 venv에 이미 설치된 torch/transformers/peft/bitsandbytes를 그대로 검증에 사용 |
| 프로세스 격리 | subprocess / multiprocessing | canary의 import·OOM 크래시가 메인 CLI를 죽이지 않도록 격리 ([ADR-0002](adr/0002-subprocess-isolation-for-canary.md)) |
| 정밀 타이밍 | torch.cuda.Event | wall-clock 대신 GPU 커널 실행 시간을 정확히 측정 |
| 패키징·배포 | PyPI (pip install) | 사용자의 기존 Python 환경에 설치돼야 그 환경을 검사할 수 있음 |
| 테스트 | pytest | 모듈별 단위 테스트, GPU 유무에 따른 조건부 테스트 |
| CI | GitHub Actions | 오픈소스 표준 CI |
| 라이선스 | MIT | 완전 permissive |

---

## 3. Canary 검증 구성 (MVP)

Canary는 독립적으로 켜고 끌 수 있는 옵션의 조합으로 구성한다. 기본 체크와 `--model` 지정 실행, 두 가지가 있다. CLI 명령어 자체의 계약은 [contracts/cli.md](contracts/cli.md) 참고.

### 기본 체크 (옵션 없이 `preflight check`)

| | |
|---|---|
| 대상 | 특정 모델과 무관 — 최소 대표 구조(예: 4bit 양자화 레이어 1~2개) |
| 목적 | GPU/드라이버/CUDA 체인이 물리적으로 살아있는가 |
| 크기 | 최소 (batch=1, seq=8 수준) |
| OOM 위험 | 사실상 없음 |
| 소요 | 수 초 |

> 이 크기는 PASS/FAIL 판정에는 문제없지만, 여기서 같이 재는 CPU 대비 실행시간 배수의 신뢰도는 별개 문제다. 크기가 작을수록 배수 신호가 흐려진다는 게 실측으로 확인됐다 (§6-01 참고).

### `--model <name> --batch-size <n> --seq-len <n>` — 모델 지정 + VRAM 실측 (MVP)

목표 모델의 실제 config를 조회해(가중치 다운로드 없음) canary를 그 구조로 구성하고, **목표 크기 그대로 딱 한 번** 실행해 VRAM을 실측한다. 외삽도 fail-fast 사전 probe도 없는 가장 단순한 형태다. 정적 계산 대신 실측을 택한 이유는 [ADR-0001](adr/0001-vram-measurement-over-static-calculation.md) 참고.

```python
result = run_canary(model, batch_size=target_batch_size, seq_len=target_seq_len)
```

- **실행 횟수**: 2회 — PyTorch 캐싱 allocator가 첫 반복에서는 안정 상태가 아니라, 1회만으로는 fragmentation을 포함한 정상 상태 메모리를 못 본다
- **OOM 위험**: 실재하지만 별도 안전장치가 필요 없다 — 프로세스 격리가 이 크래시를 그대로 잡아 정상 FAIL로 포장한다
- **CPU 대비 실행시간 비교는 하지 않는다** — 모델별 비교는 §7(향후)로 미뤘다. `--model` 실행에서는 device placement만 조회한다

### 학습 설정 세부 옵션 — MVP는 플래그 없음, 고정 가정

`--model` 실행은 QLoRA(4bit) + AdamW라는 고정 가정으로 canary를 구성한다. 사용자가 플래그로 지정하는 기능(LoRA/양자화/옵티마이저 override, 학습 스크립트 정적 파싱, unsloth 분기 실행)은 MVP 이후 확장 범위다 — 상세 우선순위는 Notion SRS 로드맵을 따른다.

---

## 4. 공통 처리 파이프라인

기본 체크·`--model` 어느 쪽이든 아래 네 단계를 동일하게 거친다.

1. **forward** — 입력을 통과시켜 출력을 얻음
2. **backward** — 오차를 역전파해 그래디언트 계산
3. **optimizer.step()** — 옵티마이저 상태(모멘텀 등)를 실제로 생성·갱신
4. **측정** — device placement · 메모리 델타 · 실행시간(CPU 대비 배수, 기본 체크에서만)

`optimizer.step()`이 모든 조합에 필수다. Adam 계열 옵티마이저는 `step()`을 한 번 호출하기 전까지 모멘텀 버퍼 등 내부 상태를 만들지 않아, forward+backward까지만 실행하면 옵티마이저 메모리 풀을 놓쳐 VRAM을 실제보다 적게 측정하게 된다.

**실행시간 측정 범위** — 기본 체크(1회)에서만 CPU 대비 배수를 잰다. `--model` 실행은 device placement 직접 조회만 한다.

**실행 횟수** — 기본 체크는 1회, `--model`의 목표 크기 실행은 2회(allocator 안정화).

---

## 5. 모듈 구성

병렬 개발을 위한 모듈 경계다. 각 모듈이 주고받는 값의 정확한 스키마(함수 시그니처)는 [contracts/canary-api.md](contracts/canary-api.md)에 별도로 관리한다 — 이 절은 각 모듈의 책임 범위만 설명한다.

### CanaryEngine
- **구현**: `src/preflight/canary/engine.py`(`run_canary_check`) · `worker.py`(subprocess 본체) · `model.py`(더미 모델/입력 구성) · `judge.py`(판정)
- **입력**: 모델명 또는 HuggingFace config (가중치 다운로드 없음)
- **처리**:
  1. `AutoConfig.from_pretrained()`로 구조만 조회(config.json, 수 KB)
  2. `from_config()`로 랜덤 초기화 모델 구성, 더미 입력 생성
  3. subprocess 격리 하에 forward+backward+`optimizer.step()` 실행
  4. device placement·메모리 델타·실행시간(CPU 폴백 대비 배수) 측정
  5. 자식 프로세스의 비정상 종료(OOM 포함)를 부모가 캐치해 정상 진단 결과로 포장
  6. FAIL 시, 텍스트 파싱이 아니라 `torch.version.cuda`·`bitsandbytes.cextension.lib.compiled_with_cuda`·`sys.version_info` 같은 안정된 파이썬 속성으로 원인을 분류한다
- **출력**: 항목별 PASS/WARN/FAIL, 실측값, 실패 시 원인과 해결 명령어

### FixExecutor
- **구현**: `src/preflight/fix/executor.py`(`suggest_fix`/`apply_fix`) · `causes.py`(원인 분류)
- **입력**: CanaryEngine이 반환한 실패 원인
- **처리**: 원인에 맞는 pip 명령어를 구성한다. 기본값은 텍스트만 출력하고, `--yes`가 명시된 경우에만 실제로 실행한다
- **출력**: 제시된(또는 실행된) 수정 명령어와 실행 로그

### ReVerifier
- **구현**: `src/preflight/reverify.py`
- **입력**: FixExecutor 실행 완료 신호
- **처리**: CanaryEngine을 동일 조건으로 재실행해 수정 전/후 결과를 비교한다
- **출력**: 수정 성공 여부 확정 (PASS로 바뀌었는지)

### 세 모듈을 잇는 진입점

세 모듈 자체는 아니지만 이들을 엮어서 CLI로 노출하는 얇은 레이어가 있다 — `src/preflight/cli.py`(Typer 진입점, `preflight check`) · `gpu.py`(pynvml GPU/드라이버 상태 조회) · `report.py`(rich 리포트·`--json` 출력). 흐름은 [contracts/canary-api.md §5.4 전체 흐름](contracts/canary-api.md)을 그대로 따른다.

---

## 6. 구현 시 주의사항

### 01 · Baseline Timing — 절대값 대신 상대 비교

GPU마다(T4, A10G, L4, A100, H100 등) 동일 연산의 물리적 소요 시간이 크게 달라, 고정된 절대 임계값("50ms 이상이면 느림")은 쓸 수 없다. 런타임에 CPU 강제 폴백 연산 시간을 1회 측정해, 그 대비 몇 배 빠른가로 판정하는 상대 비교 방식을 쓴다. 채택 배경은 [ADR-0003](adr/0003-relative-baseline-timing.md) 참고.

> **실측 결과** — bitsandbytes 4bit은 CPU에서도 정상 동작함이 확인됐다(에러 없음, 진짜 dequant 연산 경로를 탐) — 배수 비교 자체는 유효하다. 다만 canary 크기가 작으면 배수가 실제보다 작게 나오는 문제가 확인됐다 — 지금 "기본 체크" 크기(batch=1, seq=8)는 device 확인용 PASS/FAIL 판정엔 문제없지만, 배수 기반 WARN 판정 목적으로는 신뢰도가 낮을 수 있다.
>
> **구버전 bnb 대응** — CPU 강제 폴백 연산은 기본적으로 canary의 실제 연산(bnb 4bit)을 그대로 쓴다. 구버전 bnb 등으로 이 실행 자체가 실패하면 평범한 `nn.Linear`로 대체하고, 파라미터 device·`compiled_with_cuda` 메타데이터 체크를 보조 신호로 병행하며, 폴백이 발동됐다는 사실을 출력에 명시한다.

### 02 · Process Isolation — Canary 프로세스 격리

canary 실행 중 프로세스가 죽을 수 있는 경로가 두 가지 있다.

- **Import 크래시** — `import torch`, `import bitsandbytes` 자체가 CUDA 라이브러리 꼬임으로 .so 로드 실패를 일으키면 메인 프로세스가 그대로 죽을 수 있다
- **OOM 크래시** (`--model` 목표 크기 실행 시) — VRAM이 실제로 부족하면 `RuntimeError: CUDA out of memory`로 프로세스가 강제 종료된다

해결책은 동일하다 — canary는 반드시 `subprocess`로 격리 실행하고, 부모 프로세스가 자식의 비정상 종료(exit code, OOM 포함)를 캐치해 정상적인 진단 결과(FAIL)로 포장한다. 채택 배경은 [ADR-0002](adr/0002-subprocess-isolation-for-canary.md) 참고.

### 03 · Import Overhead — Import 오버헤드 현실화

기본 체크의 소요 시간을 "수 초"로 기대하기 쉬운데, 실제로는 torch/bitsandbytes 최초 import와 CUDA 컨텍스트 빌드만으로 사양에 따라 3~10초 오버헤드가 발생할 수 있다. CLI 체감 속도 기대치를 이 수치 기준으로 설정한다.

### 04 · Fix Automation Risk — Fix 자동 실행의 안전장치

conda, poetry, venv, Docker 등 가상환경 관리 방식이 사용자마다 달라, 무조건 `pip install`을 실행하면 오히려 환경을 더 꼬이게 할 위험이 있다. 환경 매니저를 감지해 그에 맞는 명령어를 만드는 기능은 향후 확장 범위로 분리돼 있다 — MVP는 이 감지 로직 없이 더 단순하게 간다.

| 범위 | 동작 |
|---|---|
| MVP | 환경 매니저 감지 없이 고정된 fix 명령어(대부분 pip 기준)를 텍스트로 제시. `--yes` 지정 시에만 그대로 실행 |
| 향후 | 위 MVP 동작 앞에 환경 매니저 감지 단계를 추가해, conda/poetry/venv 등에 맞는 명령어로 대체 제시 |

> **바뀌지 않는 것** — 감지 로직 유무와 무관하게 "기본값은 텍스트로만 제시, `--yes` 지정 시에만 실제 실행"이라는 안전장치는 MVP·향후 모두 동일하다.

**MVP 판정 항목별 FIX 방향 (초안, 미확정)**

| 탐지 항목 | 판정 | FIX 방향(안) |
|---|---|---|
| Import 크래시 | FAIL | 예외 타입별 재설치 명령. 예: bitsandbytes CUDA 미빌드 → `pip install bitsandbytes --upgrade --force-reinstall` |
| OOM(`--model` 목표 크기 실행 시) | FAIL | batch_size 축소 또는 quantization 적용 안내 |
| 4bit 레이어 device=cpu 감지 | FAIL | `compiled_with_cuda` False면 재설치 FIX, True인데도 cpu면 다른 원인으로 분기 안내 |
| CPU 대비 배수 2배 미만 | WARN | 원인 특정 어려운 회색지대 — 성능 저하 가능성 안내 수준으로 예상 |
| 메모리 델타 15% 이상 벗어남 | WARN | fragmentation·activation 재계산 등 후보 원인 안내 수준으로 예상 |

> 방향성 정리이며 최종 문구 아님.

### 05 · Attention 구현 일치

모델의 activation 메모리는 naive attention이냐 flash-attention이냐에 따라 크게 달라진다. Canary가 특정 방식으로 하드코딩돼 있으면, 실제 학습이 다른 방식으로 돌 경우 실측값이 틀어진다 — VRAM을 과대 경고하거나, 반대로 과소 추정해 실제 학습에서 OOM이 날 수 있다.

Canary는 사용자 환경에 실제로 설치·활성화된 attention 구현(`attn_implementation` 설정 등)을 그대로 반영해 모델을 구성해야 한다.

---

## 7. 향후 확장 설계 — probe 기반 외삽 + fail-fast

> MVP 이후 최우선 확장. §3의 단순 실측(목표 크기 그대로 1세트 실행)을 아래처럼 정밀화한다 — 설계는 끝났고, MVP 일정상 구현만 미룬 상태다.

`--model` 지정 시 항상 먼저 **seq=64 probe**를 1회 실행한다 — 메모리 델타와 GPU/CPU 실행시간을 이 한 번의 실행에서 함께 측정한다. probe가 FAIL이면 그 자리에서 결과를 보고하고 중단한다(fail fast).

**기본 동작(외삽)** — probe에 이어 seq=128 지점도 실측해 두 지점으로 목표 크기의 VRAM을 예측한다.

```python
mem_64  = run_canary(model, seq_len=64)   # probe — fail-fast 판정 + 시간 비교도 겸함
mem_128 = run_canary(model, seq_len=128)
per_token_cost = (mem_128 - mem_64) / (128 - 64)
predicted = mem_64 + per_token_cost * (target_seq_len - 64)
```

**`--exact`** — probe 통과 후, 외삽 대신 목표 batch_size·seq_len 그대로 추가로 확정 실행한다(근사가 아닌 확정값).

**CPU 대비 실행시간 비교 범위** — 기본 체크(1회)와 seq=64 probe(1회)에서만 측정한다. 이후 단계는 이미 probe에서 "GPU 가속 정상"이 확인된 뒤라 device 직접 조회만 한다.

얻는 이득: (1) 외삽 덕분에 OOM 위험 없이 빠르게 근사치를 얻음, (2) probe가 먼저 걸러줘서 `--exact`도 안전하게 fail-fast, (3) probe에서 GPU 가속이 확인되면 이후 단계는 재확인 없이 넘어가 비용을 아낌.

---

## 관련 문서

- CLI·모듈 API 계약: [contracts/](contracts/)
- 설계 결정의 배경("왜"): [adr/](adr/)
- 제품 기획·MVP 범위·일정·팀 구성: Notion 원본 (`TEAM_WORKFLOW.md`, SRS)
