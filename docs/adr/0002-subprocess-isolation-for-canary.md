# ADR-0002: Canary는 반드시 subprocess로 격리 실행한다

**상태**: Accepted (2026-07-30)

## 맥락

Canary 실행 중 프로세스가 죽을 수 있는 경로가 두 가지 있다.

- **Import 크래시** — `import torch`, `import bitsandbytes` 자체가 CUDA 라이브러리 꼬임으로 `.so` 로드 실패를 일으키면 그 프로세스가 그대로 죽을 수 있다.
- **OOM 크래시** — `--model` 목표 크기 실행 시 VRAM이 실제로 부족하면 `RuntimeError: CUDA out of memory`로 프로세스가 강제 종료된다.

Preflight는 진단 도구이므로, 진단 대상의 실패(OOM 등)가 진단 도구 자체를 죽여서 "결과 없음"으로 끝나서는 안 된다. 이런 크래시 자체가 유효한 진단 결과(FAIL)여야 한다.

## 결정

Canary는 항상 메인 CLI 프로세스가 아니라 별도 `subprocess`(또는 `multiprocessing`)에서 격리 실행한다. 부모 프로세스는 자식의 비정상 종료(exit code, OOM 포함)를 캐치해 정상적인 진단 결과(FAIL)로 포장한다. 원인 분류는 텍스트 파싱이 아니라 `torch.version.cuda`·`bitsandbytes.cextension.lib.compiled_with_cuda`·`sys.version_info` 같은 안정된 파이썬 속성으로 한다.

## 결과

- 어떤 크래시가 나도 CLI 자체는 항상 유효한 PASS/WARN/FAIL 결과를 반환한다.
- 프로세스 생성 오버헤드(수백 ms 수준)가 매 실행마다 발생하지만, 기본 체크 기준 이미 torch/bitsandbytes import 자체가 3~10초 걸리는 것에 비하면 무시할 수준이다.
- 이 격리 경계는 `run_canary_check()` 함수 안에 캡슐화된다 — 계약은 [contracts/canary-api.md](../contracts/canary-api.md) 참고.
