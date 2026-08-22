---
name: implement-issue
description: GitHub Issue를 분석하고 승인 후 구현, 검증, PR 생성까지 완료한다.
---

# implement-issue

GitHub Issue 하나를 읽고, 계획을 한 번 보고하고, 승인 후 구현·검증·커밋·Push·PR 생성까지 수행한다.

# 실행 전

1. 지정된 GitHub Issue를 읽는다.
2. Issue 본문에 Notion 링크가 있을 때만 해당 Notion 페이지를 읽는다. 워크스페이스 전체를 검색하지 않는다.
3. 관련 docs와 코드를 읽는다 — 코드를 만지기 전에 `docs/architecture.md`, `docs/contracts/`,
   관련 `docs/adr/`를 먼저 확인해 이미 정해진 모듈 경계·API 계약·설계 결정을 어기지 않는다.
4. 작업이 복잡하거나 장기화될 때만 `.agent/current-task.md`를 10~30줄로 작성한다(Git 미추적).
5. 요구사항, 영향 파일, 구현 계획, 테스트 계획, 위험 요소를 한 번 보고한다.
6. 사용자 승인 전에는 코드, GitHub, Notion을 수정하지 않는다.

# 승인 후 자동 수행

1. `feat/<issue-number>-<short-name>` 브랜치를 만든다.
   (종류에 따라 `fix/` · `refactor/` · `docs/` 접두사를 쓴다.)
2. 구현한다.
3. 포맷, 린트, 테스트를 실행한다.
   - `ruff format --check .`
   - `ruff check .`
   - `pytest`
   - 또는 위를 묶어서 `bash scripts/verify-before-pr.sh` (PowerShell만 쓰면 `scripts/verify-before-pr.ps1`)
4. 실패하면 원인을 분석하고 최대 2회까지 수정 후 재실행한다.
5. 검증이 통과하지 않으면 커밋과 PR 생성을 중단하고, 실패 원인·시도한 해결·사람 판단이 필요한 사항을 보고한다.
6. **(조건부) 실환경 재현 검증** — 이 Issue가 QA에서 발견되어 이슈화된 버그 픽스이거나, CI가 구조적으로 못 보는 경로
   (`@requires_cuda`/torch 관련, Windows 전용, 설치·`--yes` 자동수정·출력 가독성 관련)를 고치는 것이면 pytest만으로 끝내지 않는다.
   순수 리팩터링·문서 변경·pytest로 이미 충분히 커버되는 로직 버그는 이 단계를 생략한다.
   - `docs/정성오/qa_guide.md` 3-B/3-C 절차대로 **저장소 밖** 전용 디스포저블 venv를 새로 만들거나 완전히 지우고 다시 만든다.
     (AGENTS.md "실환경 재현 검증 예외" — 이 좁은 범위의 `pip install`/`uninstall`만 Grade D가 아닌 B로 취급된다.
     사용자의 실제 개발 venv나 다른 프로젝트 환경은 절대 건드리지 않는다.)
   - 그 venv에 저장소 경로로 `pip install`해 실제 wheel을 설치한다.
   - 원래 이슈/QA 리포트에 적힌 재현 커맨드를 그대로 실행해 수정 전 증상이 사라졌는지 확인한다.
   - 출력을 캡처해 PR 본문 "수동 확인" 항목에 첨부한다.
   - 재현이 안 되거나 새로운 이상이 보이면 5번과 동일하게 커밋·PR 생성을 중단하고 보고한다.
7. API, CLI, JSON 등 기술 계약이 변경되면 관련 docs를 **같은 PR에서** 갱신한다.
8. 장기 영향 기술 결정이 생기면 ADR 초안을 만든다(Accepted 확정은 별도 승인 사항).
9. 현재 Issue 범위 밖의 개선점은 Backlog Issue로 분리한다(label: `improvement` 등, Status: Backlog).
10. 변경 사항을 검토하고 커밋한다.
11. 원격 브랜치로 push한다.
12. PR 생성 직전, 해당 Issue의 완료 조건 체크박스 중 이번 작업으로 끝난 항목을 갱신한다.
13. PR을 생성한다. 본문은 `.github/pull_request_template.md` 구조를 따른다.
14. PR 본문에 `Closes #<issue-number>`와 테스트 결과(실환경 재현 검증을 했다면 그 결과 포함)를 포함한다.
15. 결과로 PR URL, 변경 요약, 테스트 결과(실환경 재현 검증 포함 여부와 결과), docs/ADR/Backlog 변경 사항을 보고한다.

# 자동 완료 모드

`/implement-issue #번호 자동 완료 모드`로 호출되고 사용자가 계획을 승인("진행해")하면,
위 "승인 후 자동 수행" 전체를 별도 확인 없이 끝까지 수행한다.
단, 검증이 통과하지 않으면 PR을 만들지 않는다.

# 하지 않는 것 (자동화 등급 D)

- `main` 병합, 배포
- 토큰·비밀값 변경 또는 커밋
- 사용자의 패키지·드라이버·가상환경 실제 설치·삭제·변경
  (단, 위 "실환경 재현 검증" 단계의 저장소 밖 전용 디스포저블 venv는 예외 — AGENTS.md 참고)
- Notion 페이지 삭제 또는 대규모 재구성

# Notion 정책

- Issue 본문의 연결 링크가 있을 때만 Notion을 읽는다.
- 작업 DB의 상태 및 PR/Issue 링크는 승인된 자동 작업 범위에서 갱신할 수 있다.
- 제품 목적, MVP 범위, 사용자 약속 변경은 변경안 승인 후에만 수정한다(`propose-change` 참고).
- 파일 이동, 내부 리팩터링, 단순 테스트 추가는 Notion에 기록하지 않는다.

# 사용 예시

```
/implement-issue #12 자동 완료 모드로 진행해.
```

AI가 계획을 보고하면:

```
진행해.
```
