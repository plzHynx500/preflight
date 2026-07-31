---
name: plan-feature
description: Notion 기획 또는 지정 문서를 구현 가능한 GitHub Issues로 분해한다.
---

# plan-feature

기획 문서를 "하나의 검토 가능한 PR로 끝나는" GitHub Issue 단위로 분해한다.
승인 전에는 계획 초안만 제시하고 Issue를 생성하지 않는다.

## 절차

1. 지정된 Notion 문서 또는 `docs/` 문서를 읽는다. 지정되지 않은 문서는 읽지 않는다.
2. 기능 범위, 완료 조건, 비기능 요구사항, 범위 제외를 추출한다.
3. 하나의 Issue가 하나의 검토 가능한 PR로 끝나도록 분해한다.
   - 좋은 예: `#12 Python 버전 및 가상환경 검사 구현`
   - 너무 큰 예: `#12 Preflight MVP 전체 구현`
4. 각 Issue에 제목, 목적, 완료 조건, 의존성(선행 작업), 예상 수정 파일, 위험 요소를 작성한다.
5. 먼저 계획 초안만 제시한다.
6. 사용자 승인 전에는 GitHub Issue를 생성하지 않는다.
7. 승인 후에만 GitHub Issue를 생성한다.
8. GitHub Project 추가는 Auto-add 워크플로우가 처리하므로 카드를 수동으로 중복 생성하지 않는다.
9. Issue 본문에 관련 Notion 원본 링크와 docs 링크를 포함한다.

## Issue 본문 형식

`.github/ISSUE_TEMPLATE/`의 템플릿 구조를 그대로 쓴다.

```
## 목적
## 완료 조건        (- [ ] 테스트 가능한 항목)
## 범위 제외
## 관련 문서        (Notion / docs / 관련 Issue·PR)
## 예상 영향 파일
## 위험 및 미확정 사항
```

## 라벨

- 종류: `feature` / `bug` / `improvement` / `refactor` / `docs` / `chore`
- 우선순위: `P0` / `P1` / `P2` / `P3`
- 영역: `area:cli` / `area:environment` / `area:output` / `area:docs`

## 승인 전 팀원이 확인할 것

- Issue 하나가 너무 크지 않은가?
- 각 Issue가 하나의 PR로 끝날 수 있는가?
- MVP에 불필요한 기능이 섞이지 않았는가?
- 완료 조건이 테스트 가능한가?
- 선행 Issue 순서가 맞는가?

## 사용 예시

```
/plan-feature

Notion의 "Preflight - 시스템 제안서"에서
환경 진단 MVP 부분을 GitHub Issue 단위로 분해해.
아직 Issue를 생성하지 말고 계획 초안만 보여줘.
```
