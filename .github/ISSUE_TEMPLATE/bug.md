---
name: Bug
about: 잘못 동작하는 것을 고치는 작업
title: "[Bug] "
labels: ["bug"]
---

<!--
지금 기능을 막고 있거나 데모를 실패시키는 문제라면 P0을 붙인다.
라벨은 종류(bug) 외에 우선순위(P0~P3)와 영역(area:cli / area:environment /
area:output / area:docs)을 함께 붙인다.
-->

## 목적

무엇이 어떻게 잘못 동작하는지, 왜 지금 고쳐야 하는지 한두 문장으로 작성한다.

### 재현 방법

1.
2.

### 기대 동작 / 실제 동작

- 기대:
- 실제:

### 환경

- OS (Windows / WSL / Linux):
- Python 버전:
- GPU / 드라이버 / CUDA:

## 완료 조건

- [ ] 재현 절차가 더 이상 문제를 일으키지 않는다
- [ ] 회귀 방지 테스트를 추가했다
- [ ] 테스트 가능한 완료 조건 3

## 범위 제외

- 이번 Issue에서 하지 않을 것

## 관련 문서

- Notion:
- docs:
- 관련 Issue/PR:

## 예상 영향 파일

- `src/preflight/...`
- `tests/...`

## 위험 및 미확정 사항

- 결정이 필요한 내용
