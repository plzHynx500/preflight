## Related Issue

Closes #17

## 변경 내용

- `causes.py`의 `reasons` 검사 문자열을 `judge.py`의 실제 출력 스키마(`status_import_crash`, `status_oom`, `quant_layer_device_cpu`)와 일치하도록 수정
- `executor.py`의 `suggest_fix` 단계에서 `bitsandbytes.cextension.lib.compiled_with_cuda` 값을 직접 평가해 결과에 병합하도록 기능 추가
- `tests/test_fix.py`의 모의 `judge_result()` 출력을 실제 스키마(`quant_backend`, `device`, 올바른 `reasons` 등)에 맞게 갱신

## 테스트

- [x] `ruff format --check .`
- [x] `ruff check .`
- [x] `pytest` (test_judge.py의 NotImplementedError를 제외하고 모두 통과 확인)
- [ ] 수동 확인 (필요한 경우)

## 문서 영향

- [x] 없음
- [ ] docs 계약 갱신 (`docs/contracts/`)
- [ ] ADR 추가 (`docs/adr/`)
- [ ] Notion 작업 DB 링크/상태 갱신
- [ ] Notion 기획 변경안 필요

## 제한 사항 및 후속 작업

- 없음
