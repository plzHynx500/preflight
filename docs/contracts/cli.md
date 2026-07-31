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

## 종료 코드

문제(WARN 또는 FAIL)가 하나라도 있으면 항상 0이 아닌 코드로 종료하고, 전부 PASS면 0으로 종료한다. WARN·FAIL을 서로 다른 코드로 나눌지는 세부 구현 시 정한다.

## 출력 예시

```
$ preflight check

✔ Canary 연산 실행              device=cuda · 메모리 이동 확인됨
✔ 실행 시간 12ms                CPU 대비 41배 (정상 범위)
✖ bitsandbytes 4bit 레이어      device=cpu 감지 → 조용한 CPU 폴백

FIX: pip uninstall bitsandbytes && pip install bitsandbytes --upgrade
재확인: preflight check --yes

3개 항목 확인 · 1개 문제 발견 · 소요 시간 4초
```

```
$ preflight check --model meta-llama/Llama-3.1-8B --batch-size 2 --seq-len 2048

✔ Canary 연산 실행              device=cuda · 메모리 이동 확인됨
✔ VRAM 실측                     8.4GB / 12GB 가용 — batch=2, seq=2048 그대로 실행
✔ 목표 배치 크기 적합            batch=2, seq=2048 기준

1개 모델 확인 · 문제 없음 · 소요 시간 18초
```

## 관련 문서

- 내부 모듈 간 API 계약: [canary-api.md](canary-api.md)
- 설계 배경: [../architecture.md](../architecture.md)
