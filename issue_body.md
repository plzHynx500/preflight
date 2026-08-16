## 목적 (버그 설명)

W6(#15) 리뷰 중 발견. `src/preflight/fix/causes.py`의 `classify_cause()`가 `reasons` 리스트에서 찾는 문자열이 `judge.py`가 실제로 만드는 값과 다르다.

| `causes.py`가 찾는 문자열 | `judge.py`가 실제로 내는 문자열 |
|---|---|
| `"import_crash"` | `"status_import_crash"` |
| `"oom"` | `"status_oom"` |
| `"4bit_cpu_fallback"` / `"device_cpu"` | `"quant_layer_device_cpu"` |

## 지금 당장 사용자에게 잘못된 결과가 나가진 않는 이유

`classify_cause`가 `reasons` 체크와 별개로 `status`/`quant_backend`+`device` 원본 필드도 직접 재확인하는 중복 체크를 갖고 있어서, 이 중복 체크가 실질적으로 판정을 담당하고 있다 (`reasons` 쪽 체크는 사실상 죽은 코드). 그래서 `oom`/`import_crash`/4bit-device-cpu 케이스 자체는 오늘 기준으로도 올바르게 분류된다.

## 그런데 실제로 위험한 지점 하나 발견

`_build_lines`/`classify_cause`의 4bit-cpu-fallback 분기는 `compiled_with_cuda` 키가 있어야 `"bnb_not_compiled_with_cuda"`(구체적 FIX 명령 있음)로, 없으면 `"4bit_cpu_fallback_other"`(FIX 명령 없이 안내 문구만)로 갈린다. 그런데 `compiled_with_cuda`는 `judge_result()` 출력 스키마(7개 raw 필드 + verdict + reasons)에 아예 없고, `executor.py`/`causes.py` 어디에도 이 값을 실제로 조회(`bitsandbytes.cextension.lib.compiled_with_cuda`)해서 채워 넣는 코드가 없다 — 두 파일 docstring은 이 속성으로 판별한다고 되어 있지만, 실제로 그 값을 채우는 코드가 없다. 결과적으로 이 경로가 실제로 배선되면(FixExecutor 통합 시) 4bit-cpu-fallback 케이스는 항상 `"4bit_cpu_fallback_other"`(구체적 FIX 명령 없는 쪽)로만 분류될 가능성이 높다.

## `tests/test_fix.py`도 같은 문제

기존 테스트가 실제 `judge_result()` 출력이 아니라 옛 문자열(`"import_crash"`, `"oom"`, `"4bit_cpu_fallback"`)을 손으로 만든 dict로 `classify_cause`를 호출하고 있어서, 실제 파이프라인과의 통합 오류를 못 잡는다. 특히 `test_classify_and_suggest_bnb_not_compiled`의 두 번째 케이스(`res2`)는 `quant_backend`/`device` 없이 `reasons=["4bit_cpu_fallback"]`만으로 테스트하는데, 이건 실제로는 절대 나오지 않는 조합이다.

## 완료 조건

- [x] `causes.py`의 `reasons` 체크 문자열을 `judge.py`의 실제 값(`status_oom`/`status_import_crash`/`quant_layer_device_cpu`)으로 수정 (또는 애초에 중복 체크이니 `reasons` 체크를 제거하고 `status`/`quant_backend`+`device` 직접 체크만 남기는 것도 검토)
- [x] `compiled_with_cuda`를 실제로 어디서 채울지 결정 — FixExecutor가 `bitsandbytes.cextension.lib.compiled_with_cuda`를 직접 조회해서 `classify_cause` 호출 전에 병합하는 코드 추가 필요
- [x] `tests/test_fix.py`를 `preflight.canary.judge.judge_result()`가 실제로 만드는 dict를 기준으로 재작성(적어도 통합 테스트 1~2개는 실제 judge_result() 출력을 그대로 사용)

## 범위 제외

- FixExecutor의 `--yes` 실제 실행 로직 자체 변경 없음

## 관련 문서

- docs: docs/contracts/canary-api.md (judge_result 출력 스키마), docs/architecture.md §6-04
- 관련 Issue/PR: #15/#16 (W6 리뷰 중 발견), #5/#9 (원 구현)

## 예상 영향 파일

- `src/preflight/fix/causes.py`
- `src/preflight/fix/executor.py`
- `tests/test_fix.py`

## 위험 및 미확정 사항

- `compiled_with_cuda` 조회 방식(어느 시점에 어떻게 채울지)은 FixExecutor 담당자(정성오)와 논의 필요.
