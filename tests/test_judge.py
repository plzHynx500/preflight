"""판정 기준은 docs/contracts/canary-api.md 참고."""

from preflight.canary.judge import judge_result  # noqa: F401


def test_judge_result_pass() -> None:
    raise NotImplementedError
