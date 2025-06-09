from __future__ import annotations

import sys
from pathlib import Path

from pytest_mock import MockerFixture  # type: ignore
from typer.testing import CliRunner

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.cli import app
from src.workflow import Question


def test_cli_invokes_workflow(mocker: MockerFixture) -> None:
    q = Question(reasoning="r", text="t")
    mocker.patch("src.cli.run_workflow", return_value=q)

    runner = CliRunner()
    result = runner.invoke(app, ["Will AI?"])
    assert result.exit_code == 0
    assert "reasoning='r'" in result.stdout
