# PR 생성 직전 검증 스크립트 (TEAM_WORKFLOW.md §25.3) — PowerShell 판.
# 포맷 검사 -> 린트 -> 테스트 -> 민감 파일 검사 순으로 실행한다.
# 동작은 scripts/verify-before-pr.sh 와 동일하게 유지한다.
#
# pytest 실패는 기본값으로 PR 생성을 차단한다 — .github/workflows/ci.yml이 이미
# 그렇게 동작하므로(#42/PR #50에서 continue-on-error 예외를 걷어냄) 로컬도 맞춘다.
# 급하게 우회해야 하면 PREFLIGHT_STRICT_TESTS=0으로 일시적으로 비차단 처리할 수
# 있다(-StrictTests는 반대로 명시적으로 차단을 강제한다).
#
# 포맷·린트·민감 파일 검사는 예외 없이 항상 차단 조건이다.

[CmdletBinding()]
param(
    # pytest 실패를 차단 조건으로 취급한다(기본값).
    [switch]$StrictTests = $true
)

$ErrorActionPreference = "Stop"

if ($env:PREFLIGHT_STRICT_TESTS -eq "0") { $StrictTests = $false }
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
    Write-Warning "테스트 실패 - PREFLIGHT_STRICT_TESTS=0으로 비차단 처리됨."
    Write-Warning "실제 회귀라면 PR을 만들지 말고 먼저 고칠 것."
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
