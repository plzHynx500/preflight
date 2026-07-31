# AGENTS.md

Preflight 팀의 개발 워크플로우 규칙 원본이다. Notion "TEAM_WORKFLOW.md"(v1.0, 2026-07-22)를 코드베이스에 반영한 문서이며, `CLAUDE.md`/`GEMINI.md` 등 에이전트별 설정 파일은 이 문서를 `@AGENTS.md`로 참조만 한다 — 규칙 본문은 여기 한 곳에서만 관리한다.

## 핵심 원칙

- **Notion** = 제품·기획 원본 (왜/무엇을 만들지: 문제 정의, MVP 범위, 사용자 시나리오, 조사, 회의록)
- **GitHub Issue** = 실제 개발 작업 원본 (무슨 코드 작업을 할지)
- **GitHub Project** = Issue의 시각화 보드일 뿐, 별도 정보 원본 아님
- **GitHub PR** = 실제 변경과 검증의 증거
- **docs/** = 코드와 함께 유지되는 기술 문서만 (`architecture.md`, `adr/`, `contracts/`)
- **.agent/current-task.md** = 현재 작업용 압축 브리프. Git 미추적, 복잡/장기 작업일 때만 생성
- 같은 정보를 Notion과 GitHub(문서 포함)에 이중 관리하지 않는다

## AI 에이전트가 지켜야 할 핵심 행동 규칙

1. GitHub Issue를 먼저 읽는다 → Issue 본문에 Notion 링크가 있을 때만 해당 Notion 페이지를 읽는다 → Notion 워크스페이스 전체를 매번 검색하지 않는다.
2. 코드를 만지기 전에 항상 `docs/architecture.md`·`docs/contracts/`·관련 `docs/adr/`를 먼저 읽는다 — 이미 정해진 모듈 경계·API 계약·설계 결정을 재추측하거나 어긋나게 구현하지 않기 위함이다. 기술적인 "무엇을 만들지"는 이 문서들이 원본이며 AGENTS.md에는 넣지 않는다(중복 관리 금지, 위 "핵심 원칙" 참고).
3. 작업이 복잡/장기화될 경우에만 `.agent/current-task.md`에 10~30줄로 핵심을 압축한다.
4. 코드 수정 전 항상 계획(요구사항 요약, 수정 파일, 구현 순서, 테스트 계획, 위험 요소)을 먼저 보고하고, **사용자 승인 후에만** 구현한다.
5. 승인 후에는 브랜치 생성 → 구현 → 포맷/린트/테스트 → 실패 시 최대 2회 재시도 → docs/ADR 갱신 판단 → 범위 밖 개선은 Backlog Issue로 분리 → 커밋/Push/PR 생성까지 연속 자동 수행 가능. 단, 테스트 미통과 시 PR 생성을 중단하고 원인을 보고한다.
6. PR 본문에 `Closes #번호`를 반드시 포함해 merge 시 Issue가 자동 종료되게 한다.

## 자동화 등급 — 반드시 지킬 경계

| 등급 | 범위 | 예시 |
|---|---|---|
| A | 즉시 자동 | 코드 탐색, 테스트, 린트, 포맷, `current-task.md` 생성 |
| B | 작업 계획 승인 후 자동 | 브랜치 생성, 구현, 커밋, Push, PR 생성, docs 갱신, Backlog Issue 생성 |
| C | 변경안 제시 + 별도 승인 후 자동 | Notion 기획/MVP 범위 수정, ADR Accepted 확정, 기존 Issue 완료조건 변경 |
| D | 사람만 실행, AI는 절대 자동 수행 금지 | main 병합, 배포, 토큰/비밀값 변경, 실제 패키지·드라이버·가상환경 설치/삭제/변경, Notion 페이지 삭제·대규모 재구성, 외부 공유 권한 변경 |

## Notion 운영 규칙

- Notion에 저장: 문제 정의, 목표/성공기준, MVP 범위, 사용자 시나리오, 조사자료, 회의록, 발표자료, 장기 일정/우선순위.
- Notion에 저장 안 함: 함수 내부 구현, 파일명 변경, 테스트 추가, PR별 상세 변경, Issue 세부 상태, 리팩터링 기록, 브랜치 이름 — 이런 건 GitHub/코드가 원본.
- Notion 제품 기획(MVP 범위, 사용자 약속) 수정 전에는 반드시 "Notion 변경 제안" 형식(대상/기존 내용/제안 내용/변경 이유/연결 항목)으로 변경안을 먼저 제시하고 명시적 승인("승인")을 받아야 한다. 승인 전 실제 문서 수정 금지.

## GitHub Issue/Project 규칙

- Issue 하나 = 하나의 검토 가능한 PR 크기로 분해 (예: "#12 Python 버전 및 가상환경 검사 구현" O, "#12 Preflight MVP 전체 구현" X).
- 라벨: 종류(feature/bug/improvement/refactor/docs/chore), 우선순위(P0~P3), 영역(area:cli 등).
- GitHub Project 자동 추가 워크플로우 설정 시 AI는 Issue만 생성하면 되고 Project 카드는 자동 처리된다.
- 기능 기획 → AI가 Issue 분해 초안 제시 → 팀원 승인 → Issue 생성 (초안 승인 전 Issue 생성 금지).

## docs/ · ADR 규칙

- docs/에 둘 것: `architecture.md`(구조 안정 후 갱신), `adr/`(장기 영향 결정 시), `contracts/`(CLI/API/JSON 등 여러 코드가 공유하는 형식).
- ADR은 "나중에 왜 이렇게 결정했는지" 질문이 나올 법한 결정에만 작성한다(프레임워크 선택, 아키텍처 방식 선택, 정책 변경 등). 함수명 변경/파일 이동 등에는 작성하지 않는다.

## Backlog 규칙

구현 중 발견한 범위 밖 개선점은 현재 Issue에 넣지 않고 새 GitHub Issue(label: improvement 등, Status: Backlog)로 분리한다.

## 구현 중 기획 충돌 시 처리

문제 발견 → AI가 기존 요구사항/제약/대안/영향 정리 → 팀원 승인 → Notion 원본 수정 → 관련 Issue 완료조건 수정 → (장기 결정이면) ADR 작성 → 구현/PR.

## Notion 연동

Notion MCP 연동은 토큰 기반으로 이미 구성되어 있다(OAuth 불필요). Internal Integration Token 사용 원칙(원래는 비대화형 자동화 전용)과 별개로, 이 환경에서는 대화형 작업에도 동일 토큰을 사용한다.
