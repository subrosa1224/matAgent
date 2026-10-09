"""Minimal tests for the M1 CLI skeleton."""

from typer.testing import CliRunner

from materials_screening import __version__
from materials_screening.cli import app

runner = CliRunner()


def test_version_command() -> None:
    """The version command prints the package version and exits successfully."""
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__
