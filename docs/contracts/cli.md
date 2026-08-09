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

진단 결과의 최종 판정(`verdict`)에 따라 규격화된 종료 코드를 반환한다. `--yes` 옵션으로 수정 후 재확인(`ReVerifier`) 수행 시 1차 판정이 아닌 재확인 최종 결과 기준의 종료 코드를 반환한다.

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

## 관련 문서

- 내부 모듈 간 API 계약: [canary-api.md](canary-api.md)
- 설계 배경: [../architecture.md](../architecture.md)
