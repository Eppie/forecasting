from __future__ import annotations

import sys
from pathlib import Path

from pytest_mock import MockerFixture  # type: ignore
from typer.testing import CliRunner

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.cli import app


def test_cli_invokes_workflow(mocker: MockerFixture) -> None:
    mocker.patch("src.cli.run_workflow", return_value={"probability": 0.42, "rationale": "r"})

    runner = CliRunner()
    result = runner.invoke(app, ["Will AI?"])
    assert result.exit_code == 0
    assert "0.42" in result.stdout
