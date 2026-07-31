from typer.testing import CliRunner

from preflight.cli import app

runner = CliRunner()


def test_check_command_exists() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
