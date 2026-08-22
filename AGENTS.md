# AGENTS.md

Preflight 팀의 개발 워크플로우 규칙 원본이다. Notion "TEAM_WORKFLOW.md"(v1.0, 2026-07-22)를 코드베이스에 반영한 문서이며, `CLAUDE.md`/`GEMINI.md` 등 에이전트별 설정 파일은 이 문서를 `@AGENTS.md`로 참조만 한다 — 규칙 본문은 여기 한 곳에서만 관리한다. (현재 repo에는 `CLAUDE.md`만 있고 `GEMINI.md`는 없다. Antigravity CLI 팀원이 생기면 같은 `@AGENTS.md` 한 줄짜리로 추가한다.)

## 프로젝트 목적

Preflight는 로컬 AI·파인튜닝 환경의 설치 가능 여부를 진단하고, 문제 원인과 안전한 해결 절차를 제안하는 CLI 도구다.

## 공통 원칙

- 요청 범위를 벗어난 리팩터링을 하지 않는다.
- 불확실한 요구사항은 추측해 구현하지 않는다.
- 코드 수정 전 관련 GitHub Issue와 필요한 docs를 확인한다.
- 사용자의 패키지, 드라이버, 가상환경을 동의 없이 변경하지 않는다.
- 토큰, API 키, `.env` 파일, 개인 경로를 커밋하지 않는다.

## 테스트 명령

- 테스트: `pytest`
- 린트: `ruff check .`
- 포맷: `ruff format .`

## 핵심 원칙 — 도구별 역할

한 줄 원칙: **Notion은 "왜/무엇을 만들지", GitHub는 "무슨 코드 작업을 할지", 로컬 docs는 "코드가 따라야 할 기술 약속", PR은 "실제로 무엇을 바꿨는지"를 기록한다.**

- **Notion** = 제품·기획 원본 (문제 정의, 목표/성공기준, MVP 범위, 사용자 시나리오, 조사, 회의록, 발표 자료). 제품 요구가 바뀔 때만 갱신.
- **GitHub Issue** = 실제 개발 작업 원본 (기능/버그/테스트/리팩터링/문서/조사/후속 개선). 작업 중 지속 갱신.
- **GitHub Project** = Issue의 시각화 보드일 뿐, 별도 정보 원본 아님.
- **GitHub PR** = 실제 변경과 검증의 증거.
- **docs/** = 코드와 함께 유지되는 기술 문서만 (`architecture.md`, `adr/`, `contracts/`). 기술 계약이 바뀔 때만 갱신.
- **.agent/current-task.md** = 현재 작업용 압축 브리프. Git 미추적, 복잡/장기 작업일 때만 생성.
- **AGENTS.md** = 에이전트 공통 행동 규칙. 팀 규칙이 바뀔 때만 갱신.
- 같은 정보를 Notion과 GitHub(문서 포함)에 이중 관리하지 않는다.

## 하지 않을 것

- Notion의 긴 기획 문서를 `docs/`에 통째로 복사하지 않는다.
- Notion 작업 DB와 GitHub Issue를 같은 세부 Todo 목록으로 이중 관리하지 않는다.
- 모든 작은 코드 변경에 ADR을 작성하지 않는다.
- 모든 Issue마다 `current-task.md`를 강제하지 않는다.
- 구현 세부사항(함수명 변경, 파일 이동 등)을 Notion에 기록하지 않는다.
- Notion 워크스페이스 전체를 매 작업마다 검색하지 않는다.
- 기획 변경이 생겼는데 코드만 먼저 바꾸고 원본 기획과 Issue를 방치하지 않는다.

## AI 에이전트가 지켜야 할 핵심 행동 규칙

1. GitHub Issue를 먼저 읽는다 → Issue 본문에 Notion 링크가 있을 때만 해당 Notion 페이지를 읽는다 → Notion 워크스페이스 전체를 매번 검색하지 않는다.
2. 코드를 만지기 전에 항상 `docs/architecture.md`·`docs/contracts/`·관련 `docs/adr/`를 먼저 읽는다 — 이미 정해진 모듈 경계·API 계약·설계 결정을 재추측하거나 어긋나게 구현하지 않기 위함이다. 기술적인 "무엇을 만들지"는 이 문서들이 원본이며 AGENTS.md에는 넣지 않는다(중복 관리 금지, 위 "핵심 원칙" 참고).
3. 작업이 복잡/장기화될 경우에만 `.agent/current-task.md`에 10~30줄로 핵심을 압축한다.
4. 코드 수정 전 항상 계획(요구사항 요약, 수정 파일, 구현 순서, 테스트 계획, 위험 요소와 미확정 사항)을 먼저 보고하고, **사용자 승인 후에만** 구현한다. 승인 전에는 코드·GitHub·Notion을 수정하지 않는다.
5. 승인 후에는 브랜치 생성 → 구현 → 포맷/린트/테스트 → 실패 시 최대 2회 재시도 → docs/ADR 갱신 판단 → 범위 밖 개선은 Backlog Issue로 분리 → 커밋/Push/PR 생성까지 연속 자동 수행 가능. 단, 테스트 미통과 시 커밋·PR 생성을 중단하고 실패 원인·시도한 해결·사람 판단이 필요한 사항을 보고한다.
6. PR 생성 직전, 해당 Issue의 완료 조건 체크박스 중 이번 작업으로 끝난 항목을 갱신한다 — GitHub가 자동으로 체크해주지 않으므로, 안 하면 팀원이 이슈만 보고는 진행 상황을 알 수 없다.
7. PR 본문에 `Closes #번호`를 반드시 포함해 merge 시 Issue가 자동 종료되게 한다.
8. 작업 완료 보고에는 변경 파일, 요구사항 충족 여부, 실행한 테스트와 결과, 남은 제한 사항, 새로 발견한 개선점을 포함한다.

## 자동화 등급 — 반드시 지킬 경계

원칙: **반복적이고 되돌릴 수 있는 작업은 자동화하고, 제품 범위·외부 시스템·배포에 영향을 주는 작업은 명확한 승인 지점 뒤에 자동화한다.**

| 등급 | 범위 | 예시 |
|---|---|---|
| A | 즉시 자동 | 코드 탐색, 테스트, 린트, 포맷, `current-task.md` 생성, Issue/연결된 Notion 문서 읽기 |
| B | 작업 계획 승인 후 자동 | 브랜치 생성, 구현, 커밋, Push, PR 생성, docs 계약 갱신, Backlog Issue 생성, 승인된 Notion 작업 DB의 상태·PR/Issue 링크 갱신, (조건부) 전용 디스포저블 venv에서의 실환경 재현 검증 — 아래 "실환경 재현 검증 예외" 참고 |
| C | 변경안 제시 + 별도 승인 후 자동 | Notion 기획/MVP 범위·사용자 약속 수정, ADR Accepted 확정, 기존 Issue 완료조건 변경, 회의 요약·보고 페이지 생성 |
| D | 사람만 실행, AI는 절대 자동 수행 금지 | main 병합, 배포, 토큰/비밀값 변경, 실제 패키지·드라이버·가상환경 설치/삭제/변경(전용 디스포저블 검증 venv 예외 있음 — 아래 참고), Notion 페이지 삭제·대규모 재구성, 외부 공유 권한 변경 |

Issue 자동 종료(PR merge 시)와 Project Done 이동은 GitHub 자동화가 처리한다 — AI가 수동으로 하지 않는다.

### 실환경 재현 검증 예외 (D → B)

원칙적으로 실제 패키지·가상환경 설치/삭제는 Grade D(사람만 실행)다. 단, 아래 조건을 모두 만족하는 `pip install`/`uninstall`은 예외로 **Grade B(계획 승인 후 자동)**로 취급한다:

- 저장소 **밖**의 전용 검증 venv에서만 실행한다 (`docs/정성오/qa_guide.md` 3-B/3-C 절차 참고 — 예: 고정 경로에 매번 재생성).
- 그 venv는 매 검증마다 새로 만들거나 완전히 지우고 다시 만드는 **일회성·디스포저블** 용도로만 쓴다.
- 사용자의 실제 개발 venv(A), 기존에 쓰던 다른 프로젝트 환경, 시스템 전역 설치, 드라이버는 이 예외에 포함되지 않는다 — 여전히 D.

**적용 대상**: QA에서 발견되어 이슈화된 버그 픽스, 또는 CI가 구조적으로 못 보는 경로(`@requires_cuda`/torch 관련, Windows 전용, 설치·`--yes` 자동수정·출력 가독성 관련)를 고치는 이슈. 순수 리팩터링·문서 변경·pytest로 이미 충분히 커버되는 로직 버그는 생략한다.

절차와 실패 시 처리는 `implement-issue` 스킬 "승인 후 자동 수행" 단계를 따른다.

## 자동 완료 정책

사용자가 `/implement-issue #번호 자동 완료 모드`를 요청하고 구현 계획을 승인("진행해")하면, 에이전트는 별도 확인 없이 위 5번 흐름(브랜치 → 구현 → 검증 → docs/ADR 판단 → Backlog 분리 → 커밋/Push/PR)을 끝까지 수행한다. 팀원이 입력하는 프롬프트는 원칙적으로 "지시 한 번 + 승인 한 번"이면 충분해야 한다. 검증이 통과하지 않으면 PR을 만들지 않는다.

## Notion 운영 규칙

- Notion에 저장: 문제 정의, 목표/성공기준, MVP 범위와 범위 제외, 사용자 시나리오·UX 기획, 시장/경쟁 조사, 회의록과 역할 분담, 발표·데모 자료, 장기 일정과 상위 우선순위.
- Notion에 저장 안 함: 함수 내부 구현, 파일명 변경, 테스트 추가, PR별 상세 변경, Issue 세부 상태, 리팩터링 기록, 브랜치 이름 — 이런 건 GitHub/코드가 원본.
- 읽기 기준: Issue → (링크가 있을 때만) Notion → 필요한 부분만 `current-task.md`로 압축 → 구현·테스트 중에는 Notion을 반복 조회하지 않는다. 긴 페이지는 전체 복사 대신 필요한 섹션만 읽는다.
- 쓰기 기준: 조회는 자동, 작업 DB의 PR/Issue 링크·허용된 상태값 갱신은 작업 승인 후 자동(대상 DB/속성이 명확할 때만), 회의 요약·보고 페이지 생성과 MVP 범위·사용자 약속 수정은 변경안 승인 후 자동, 기존 기획 대량 삭제·재구성은 자동 금지.
- Notion 제품 기획(MVP 범위, 사용자 약속) 수정 전에는 반드시 "Notion 변경 제안" 형식(대상 / 기존 내용 / 제안 내용 / 변경 이유 / 연결 항목)으로 변경안을 먼저 제시하고 명시적 승인("승인")을 받아야 한다. 승인 전 실제 문서 수정 금지.

### Notion 연동 현황

TEAM_WORKFLOW.md의 기본값은 공식 Hosted Notion MCP(`https://mcp.notion.com/mcp`, OAuth)이고, Internal Integration Token은 CI·Webhook 같은 비대화형 자동화에만 추가하도록 규정한다. 다만 이 환경에서는 이미 토큰 기반 커스텀 MCP가 구성되어 있어(OAuth 불필요) 대화형 작업에도 동일 토큰을 사용한다 — 위 읽기/쓰기 기준은 연결 방식과 무관하게 동일하게 적용한다.

토큰은 GitHub Secrets, OS Secret Manager, CI Secret Store에만 보관한다. Git 저장소 커밋, `AGENTS.md`/`CLAUDE.md`/`GEMINI.md` 기록, 팀 채팅, PR 본문, 스크린샷 공유는 절대 금지다.

## GitHub Issue 규칙

- Issue 하나 = 하나의 검토 가능한 PR 크기로 분해 (예: "#12 Python 버전 및 가상환경 검사 구현" O, "#12 Preflight MVP 전체 구현" X).
- 기능 기획 → AI가 Issue 분해 초안 제시(제목/목적/완료 조건/선행 작업/예상 수정 파일/위험 요소/범위 제외) → 팀원 승인 → Issue 생성. 초안 승인 전 Issue 생성 금지.
- 승인 전 팀원이 확인할 것: Issue가 너무 크지 않은가, 하나의 PR로 끝나는가, MVP에 불필요한 기능이 섞이지 않았는가, 완료 조건이 테스트 가능한가, 선행 Issue 순서가 맞는가.

Issue 본문 구조:

```
## 목적
## 완료 조건        (- [ ] 테스트 가능한 항목)
## 범위 제외
## 관련 문서        (Notion / docs / 관련 Issue·PR)
## 예상 영향 파일
## 위험 및 미확정 사항
```

라벨 체계:

- 종류: `feature` / `bug` / `improvement` / `refactor` / `docs` / `chore`
- 우선순위: `P0` 지금 막힘·즉시 해결 / `P1` 이번 MVP 필수 / `P2` 중요하지만 MVP 후 또는 여유 시 / `P3` 아이디어 보관
- 영역: `area:cli` / `area:environment` / `area:output` / `area:docs`

"무슨 종류의 작업인가"는 라벨로, "어디까지 진행됐는가"는 Project의 Status로 관리한다.

## GitHub Project 규칙

- Status 흐름: Backlog → Ready → In Progress → In Review → Done. 필드: Status, Priority(P0~P3), Area(CLI/Environment/Output/Docs), Iteration(필요할 때만).
- "Project 카드 생성" = Issue를 만들면 보드에 표시되는 것을 뜻한다. Notion에 별도 카드·작업 페이지를 만들라는 뜻이 아니다.
- Auto-add 워크플로우(대상 repo의 open Issue 중 feature/bug/improvement/refactor 라벨 → Backlog 자동 추가)를 설정하면 AI는 Issue만 생성하면 되고 Project 카드는 자동 처리된다.

## 브랜치·PR 규칙

브랜치 이름 (Notion 원본 표기 그대로 — 라벨 이름과 접두사가 다른 것은 의도된 것이다):

```
feat/12-cli-result-renderer
fix/27-gpu-detection-windows
refactor/31-diagnosis-result-model
docs/18-safe-fix-policy
```

PR 본문:

```
## Related Issue
Closes #12

## 변경 내용
- 변경 항목

## 테스트
- [x] ruff format --check .
- [x] ruff check .
- [x] pytest
- [x] 수동 확인 (필요한 경우)

## 문서 영향
- [ ] 없음 / docs 계약 갱신 / ADR 추가 / Notion 링크·상태 갱신 / Notion 기획 변경안 필요

## 제한 사항 및 후속 작업
-
```

## docs · ADR 규칙

- docs/에 둘 것: `architecture.md`(구조 안정 후 갱신), `adr/`(장기 영향 결정 시), `contracts/`(CLI/API/JSON 등 여러 코드가 공유하는 형식).
- docs/에 두지 않을 것: Notion 기획 복사본, 회의록, 경쟁 서비스 조사, 진행률, 작업 할당, 작은 코드 변경 기록, Issue별 진행 메모.
- 갱신 기준: 내부 함수 리팩터링·Windows 전용 버그 수정 → docs 변경 없음(PR/Issue에만 기록). JSON 필드 구조나 CLI 명령 이름 변경 → 해당 계약 문서 수정. 제공 기능 범위 변경·MVP 범위 축소 → Notion + Issue + (필요 시) ADR.
- 코드 계약이 바뀌면 docs도 **같은 PR에서** 갱신한다.
- ADR은 "나중에 왜 이렇게 결정했는지" 질문이 나올 법한 결정에만 작성한다(CLI 프레임워크 선택, 결과 JSON 스키마 통일, 자동 수정 대신 명령 제안, 저장소 선택, GPU 검사 범위 제한, OS별 지원 범위 차등 등). 함수명 변경·파일 이동·색상 변경·단순 테스트 추가·중복 제거에는 작성하지 않는다.
- ADR 형식: 제목, Status / Date / Related Issue, Context, Decision, Alternatives, Consequences. ADR은 코드와 함께 Git으로 관리한다.

## .agent/current-task.md 규칙

- 다음일 때만 만든다: 작업이 하루 이상 이어질 가능성, 연결된 Notion 기획이 긺, 여러 에이전트가 같은 Issue를 이어받음, 세션 중단 후 재개, 구현 제약·완료 조건을 확실히 고정해야 함. 단순 수정·한 파일 변경·30분 내 작업에는 만들지 않는다.
- 절차: Issue 읽기 → (링크 있을 때만) Notion 읽기 → docs·코드 탐색 → 필요한 내용만 10~30줄 정리 → 이후 Notion 반복 조회 금지.
- 담을 내용: 목표, 완료 조건, 고정 제약, 관련 Issue/Notion/ADR 링크. Git 미추적(`.gitignore`에 `.agent/` 등록됨).

## Backlog 규칙

- 구현 중 발견한 범위 밖 개선점은 현재 Issue에 넣지 않고 새 GitHub Issue(label: improvement 등, 우선순위 라벨, Project Status: Backlog)로 분리한다. 현재 Issue는 원래 완료 조건까지만 끝낸다.
- Backlog Issue 본문: 현재 동작 / 문제 / 제안 / 사용자 영향 / 지금 처리하지 않는 이유 / 관련 Issue·PR.
- 정리 시점: 기능 단위 구현 직후, 주간 회의 전후, 다음 Issue 선택 전, 데모·발표 범위 확정 전. 확인할 것: Ready로 못 옮긴 P0/P1, MVP에 불필요한 P2/P3, 닫을 Issue, 합칠 Issue, 다음 구현 단위.

## 구현 중 기획 변경

판단 기준:

| 상황 | 처리 |
|---|---|
| 내부 구현만 더 나은 방식으로 변경 | PR에 기록하고 구현 |
| 현재 Issue의 완료 조건을 충족할 수 없음 | Issue에서 변경 제안 |
| 사용자가 받는 기능 범위가 바뀜 | Notion 기획 + Issue 수정 |
| 장기 기술 정책이 바뀜 | ADR 추가 |
| 나중에 하면 좋은 개선 발견 | 새 GitHub Issue → Backlog |
| 보안·실행 불가·데모 실패 문제 | 즉시 수정 또는 P0 Bug Issue |

절차: 문제 발견 → AI가 기존 요구사항/문제/대안(최소 3가지)/영향 정리 → 팀원 승인 → Notion 원본 수정 → 관련 Issue 완료조건 수정 → (장기 결정이면) ADR 작성 → 구현/PR. 코드만 조용히 바꾸지 않는다.

변경 제안 형식(Issue 댓글): 기존 요구사항 / 구현 검토 결과 / 제안 / 영향.

## 빠른 판단표

| 질문 | 해야 할 일 |
|---|---|
| 한 PR로 끝나는 단순 수정인가? | Issue 확인 후 바로 구현 |
| 기획 원문이 필요한가? | Issue 링크의 Notion 페이지만 읽기 |
| 작업이 길거나 여러 AI가 이어받는가? | `current-task.md` 생성 |
| 코드가 따라야 할 형식이 생겼는가? | `docs/contracts/` 작성 |
| 중요한 기술 선택이 생겼는가? | ADR 작성 |
| 제품 약속이 바뀌는가? | Notion + Issue 수정 (승인 후) |
| 나중에 하면 좋은 개선인가? | Backlog Issue 생성 |
| 현재 기능을 막는 문제인가? | 현재 Issue 또는 P0 Bug로 처리 |

## 최종 체크리스트

시작 전: Issue가 한 PR 크기인가 / 완료 조건과 범위 제외가 있는가 / 필요한 경우에만 Notion 링크가 연결돼 있는가 / Project에서 우선순위와 상태를 확인했는가.

구현 중: 범위 밖 개선점을 Backlog로 분리했는가 / 기획 변경을 승인 없이 코드만 바꾸지 않았는가 / ADR 필요 여부를 검토했는가 / 긴 작업이면 `current-task.md`를 최신화했는가.

PR 전: 포맷·린트·테스트를 실행했는가 / Issue 완료 조건 체크박스를 갱신했는가 / 본문에 `Closes #번호`를 넣었는가 / 변경 내용과 테스트 결과를 적었는가 / 계약이 바뀌면 docs를 같은 PR에서 갱신했는가 / 제품 범위가 바뀌면 Notion 원본도 갱신했는가.

## AI 에이전트 스킬

반복 프롬프트는 프로젝트 스킬(`SKILL.md`를 포함한 폴더)로 만든다. TEAM_WORKFLOW.md가 규정한 스킬 셋을 `.agents/skills/`에 파일로 만들어뒀다.

- `.agents/skills/plan-feature/SKILL.md` — 기획 문서를 구현 가능한 GitHub Issue 단위로 분해하고, 승인 전에는 계획 초안만 제시한다.
- `.agents/skills/implement-issue/SKILL.md` — Issue 분석 → 계획 보고 → 승인 후 구현·검증·커밋·Push·PR 생성까지 수행한다.
- `.agents/skills/propose-change/SKILL.md` — 구현 중 요구사항 충돌·위험·범위 변경을 발견했을 때 변경안을 작성한다.

**주의** — 이 경로는 TEAM_WORKFLOW.md(Antigravity CLI 기준) 원본 표기를 그대로 따른 것이다. Claude Code가 이 폴더를 자체 슬래시 커맨드로 자동 인식하는지는 확인되지 않았다 — 세션 시작 시 노출되는 스킬 목록에 아직 나타나지 않는다. 안 되면 사람이나 에이전트가 파일 내용을 직접 참고해서 그 절차를 따르면 되고, 자동 인식 여부와 무관하게 문서로서는 유효하다.

## 자동화 누락 방지 구조

AGENTS.md만으로는 부족하다. 긴 구현·디버깅이 이어지면 마무리 절차가 빠질 수 있으므로 네 계층을 함께 쓴다.

| 계층 | 역할 | 현재 상태 |
|---|---|---|
| `AGENTS.md` | 팀 규칙과 안전 경계 | 있음 (이 문서) |
| `.agents/skills/` | 작업별 순서와 체크리스트 | 있음 (위 참고) |
| `.agents/hooks.json` + `scripts/verify-before-pr.sh`(`.ps1`) | 포맷·린트·테스트·민감 파일 검사의 기계적 강제 | 있음 — 단 `.agents/hooks.json`은 Antigravity CLI 전용 형식이라 Claude Code에서는 아무 효과가 없다(Claude Code는 `.claude/settings.json`에 별도 hooks 스키마를 쓰며, 이 repo엔 아직 없다). Claude Code 세션에서는 `implement-issue` 스킬의 검증 단계나 `scripts/verify-before-pr.*`을 직접 호출하는 것으로 대체한다 |
| `.github/workflows/ci.yml` + branch protection | 병합 전 최종 안전망 | CI 워크플로 있음. branch protection도 이미 켜져 있다 — main에 `verify (3.9)`/`verify (3.12)`가 required status check로 걸려 있다(아래 참고) |

`verify-before-pr`는 포맷 검사 → 린트 → 테스트 → 민감 파일(`.env`, credentials, secret) 검사 순으로 실행한다. **현재는 pytest 단계만 예외적으로 비차단이다** — `tests/`에 구현 전 골격 스텁(`raise NotImplementedError`)이 남아 있어, 지금 무조건 차단으로 두면 문서만 고치는 PR까지 100% 막혀 검증 자체가 무력화되기 때문이다(포맷·린트·민감 파일 검사는 지금도 예외 없이 차단). `--strict-tests`(`-StrictTests`, 또는 `PREFLIGHT_STRICT_TESTS=1`)로 즉시 강제할 수 있고, 첫 구현 Issue가 머지되어 스텁이 사라지면 이 예외를 없애고 기본값을 strict로 되돌린다. `.github/workflows/ci.yml`의 Test 스텝도 같은 이유로 `continue-on-error: true`다 — 포맷·린트는 CI에서도 지금부터 실효 있는 차단 조건이다.

Hook은 편의 장치이지 CI를 대체하는 최종 안전망이 아니며, 불안정하면 Hook을 끄고 스킬 + CI로 돌아간다.

### 아직 이 repo에 없는 것 (사람이 해야 함)

- **`main` branch protection**은 이미 켜져 있다(required status check: `verify (3.9)`, `verify (3.12)`) — 이 항목은 더 이상 "없는 것"이 아니다. 단, 위 pytest 예외 때문에 지금은 포맷·린트만 실질적인 게이트다. CI 매트릭스에 job을 추가할 때 기존 job 이름(`verify (버전)`)이 바뀌면 required check가 깨지므로 주의(#49에서 `verify-windows`를 별도 job으로 분리해 회피한 이유). 새 job을 required로 승격할지는 여전히 사람이 branch protection 설정에서 결정해야 한다.
- **GitHub Project 보드**("Preflight MVP")와 Auto-add 워크플로우 — 생성 안 함.
- **`gh auth refresh -s project`** — 현재 `gh` 토큰에 `project` 스코프가 없어 Project 관련 CLI 조작이 막혀 있다. 스코프 확장은 토큰 권한 변경이라 자동화 등급 D(사람만 실행)로 분류해 시도하지 않았다. `gh`는 이미 로그인돼 있다(Issue 생성 등에는 지금도 문제없이 쓸 수 있다) — Project 조작에만 추가 스코프가 필요하다.

이 세 가지를 만들지 켤지는 팀 판단이 필요해 별도로 남겨둔다.
