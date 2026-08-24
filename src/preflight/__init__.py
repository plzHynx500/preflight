"""Preflight — 파인튜닝을 시작하기 전, 지금 환경이 실제로 준비됐는지 확인합니다."""

#: `pyproject.toml`의 `version`과 **같아야 한다.** 여기 두는 이유는 런타임에
#: `importlib.metadata`를 읽으면 소스 트리에서 실행할 때 "지금 설치돼 있는 다른
#: 버전"이 나와 오히려 헷갈리기 때문이다 — 이 트리가 그 버전이라는 보장이 없다.
#:
#: 대신 두 값이 어긋나면 `tests/test_cli.py::test_version_matches_pyproject`가
#: 깨진다. 0.1.1을 찍을 때 `pyproject`만 올려서 `--version`이 0.1.0을 출력하는
#: 것을 발견하고 묶었다(#183) — 그때까지 아무도 두 값을 대조하지 않았다.
__version__ = "0.1.1"
