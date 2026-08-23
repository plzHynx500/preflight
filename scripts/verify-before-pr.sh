#!/usr/bin/env bash
# PR 생성 직전 검증 스크립트 (TEAM_WORKFLOW.md §25.3).
# 포맷 검사 → 린트 → 테스트 → 민감 파일 검사 순으로 실행한다.
# WSL / Git Bash 기준. PowerShell만 쓰는 팀원은 scripts/verify-before-pr.ps1 을 쓴다.
#
# pytest 실패는 기본값으로 PR 생성을 차단한다 — `.github/workflows/ci.yml`이 이미
# 그렇게 동작하므로(#42/PR #50에서 continue-on-error 예외를 걷어냄) 로컬도 맞춘다.
# 급하게 우회해야 하면 `PREFLIGHT_STRICT_TESTS=0`으로 일시적으로 비차단 처리할 수
# 있다(`--strict-tests`는 반대로 명시적으로 차단을 강제한다 — 기본값과 같지만
# env var로 이미 0이 설정된 상황을 다시 덮어쓸 때 쓴다).
#
# 포맷·린트·민감 파일 검사는 예외 없이 항상 차단 조건이다.

set -euo pipefail

STRICT_TESTS="${PREFLIGHT_STRICT_TESTS:-1}"
for arg in "$@"; do
  case "$arg" in
    --strict-tests) STRICT_TESTS=1 ;;
    -h|--help)
      echo "usage: $0 [--strict-tests]"
      echo "  --strict-tests   pytest 실패를 차단 조건으로 취급 (기본값. PREFLIGHT_STRICT_TESTS=0으로 일시 해제 가능)"
      exit 0
      ;;
    *)
      echo "unknown option: $arg" >&2
      exit 2
      ;;
  esac
done

# 저장소 루트에서 실행되도록 고정 (scripts/ 안에서 호출해도 동작하게)
cd "$(git rev-parse --show-toplevel)"

echo "[1/4] Format check"
ruff format --check .

echo "[2/4] Lint"
ruff check .

echo "[3/4] Test"
if pytest; then
  echo "pytest OK"
else
  if [ "$STRICT_TESTS" = "1" ]; then
    echo "테스트 실패 — PR 생성을 중단한다." >&2
    exit 1
  fi
  echo "" >&2
  echo "!! 테스트 실패 — PREFLIGHT_STRICT_TESTS=0으로 비차단 처리됨." >&2
  echo "!! 실제 회귀라면 PR을 만들지 말고 먼저 고칠 것." >&2
  echo "" >&2
fi

echo "[4/4] Sensitive file check"
# TEAM_WORKFLOW.md §25.3의 정규식을 그대로 쓰되 두 군데를 고쳤다.
#  1) `git status --short`는 각 줄 앞에 3글자 상태 접두사(" M ", "?? ")를 붙이므로
#     원본의 `(^|/)\.env` 는 저장소 루트의 `.env` 를 잡지 못한다 → cut 으로 접두사 제거.
#  2) `--untracked-files=all` 이 없으면 새 디렉터리가 "dir/" 하나로 접혀 그 안의
#     credentials 파일이 안 보인다.
#  3) 대소문자 무시(-i)로 SECRET / Credentials 도 잡는다.
if git status --short --untracked-files=all | cut -c4- \
    | grep -iE '(^|/)\.env($|\.|/)|credentials|secret'; then
  echo "Potential secret file detected. Review before creating PR." >&2
  exit 1
fi

echo "Verification passed."
