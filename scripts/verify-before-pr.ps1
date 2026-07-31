# PR 생성 직전 검증 스크립트 (TEAM_WORKFLOW.md §25.3) — PowerShell 판.
# 포맷 검사 -> 린트 -> 테스트 -> 민감 파일 검사 순으로 실행한다.
# 동작은 scripts/verify-before-pr.sh 와 동일하게 유지한다.
#
# ─────────────────────────────────────────────────────────────────────────────
# [테스트 단계 정책 — 지금은 의도적으로 "비차단"이며, 곧 되돌린다]
#
# 현재 tests/ 에는 구현 전 골격 스텁이 있다(tests/test_engine.py 등이
# `raise NotImplementedError`). 실수가 아니라 의도된 상태다 —
# docs/architecture.md §5에서 모듈 경계와 계약은 확정했고 구현만 미착수다.
# 따라서 "pytest 실패 = 무조건 차단"으로 두면 지금은 문서만 고치는 PR까지 포함해
# 100% 확률로 막히고, 팀원이 이 스크립트를 습관적으로 건너뛰게 된다.
# 그래서 실패를 숨기지 않고 화면에 남기되(무시가 아니라 경고) 차단은 하지 않는다.
#
# 되돌리는 시점: 첫 구현 Issue가 머지되어 NotImplementedError 스텁이 사라지는 PR에서
# 기본값을 strict 로 올리고 이 주석 블록을 지운다.
# 그 전이라도 -StrictTests 또는 PREFLIGHT_STRICT_TESTS=1 로 즉시 강제할 수 있다.
#
# 포맷·린트·민감 파일 검사는 지금도 예외 없이 차단 조건이다.
# ─────────────────────────────────────────────────────────────────────────────

[CmdletBinding()]
param(
    # pytest 실패를 차단 조건으로 취급한다.
    [switch]$StrictTests
)

$ErrorActionPreference = "Stop"

if ($env:PREFLIGHT_STRICT_TESTS -eq "1") { $StrictTests = $true }

# 저장소 루트에서 실행되도록 고정
Set-Location (git rev-parse --show-toplevel)

# 주의: $ErrorActionPreference = "Stop" 은 네이티브 exe의 종료 코드를 잡지 못한다.
# ruff / pytest 는 예외를 던지지 않고 exit code만 남기므로 매번 직접 확인해야 한다.

Write-Output "[1/4] Format check"
ruff format --check .
if ($LASTEXITCODE -ne 0) { throw 'Format check failed. ruff format . 을 실행할 것.' }

Write-Output "[2/4] Lint"
ruff check .
if ($LASTEXITCODE -ne 0) { throw "Lint failed." }

Write-Output "[3/4] Test"
pytest
if ($LASTEXITCODE -ne 0) {
    if ($StrictTests) {
        throw "테스트 실패 - PR 생성을 중단한다."
    }
    Write-Warning "테스트 실패 - 현재는 차단하지 않는다 (tests/ 에 미구현 스텁이 남아 있음)."
    Write-Warning "위 실패가 스텁이 아니라 실제 회귀라면 PR을 만들지 말고 먼저 고칠 것."
    Write-Warning "강제로 차단하려면: ./scripts/verify-before-pr.ps1 -StrictTests"
}

Write-Output "[4/4] Sensitive file check"
# .sh 판과 같은 이유로 원본 정규식을 보정했다.
#  1) `git status --short` 의 3글자 상태 접두사를 제거해야 루트의 .env 도 잡힌다.
#  2) --untracked-files=all 이 없으면 새 디렉터리가 한 줄로 접혀 내부 파일을 놓친다.
$changed = git status --short --untracked-files=all | ForEach-Object { $_ -replace '^.{3}', '' }
$dangerous = $changed | Select-String -Pattern '(^|/)\.env($|\.|/)|credentials|secret'
if ($dangerous) {
    $dangerous | ForEach-Object { Write-Output $_.Line }
    throw "Potential secret file detected. Review before creating PR."
}

Write-Output "Verification passed."
