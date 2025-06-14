from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from typer.testing import CliRunner

requests_stub = ModuleType("requests")
requests_stub.get = lambda *args, **kwargs: None  # type: ignore[attr-defined]
sys.modules.setdefault("requests", requests_stub)

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.cli import app  # noqa: E402


def test_cli_invokes_workflow() -> None:
    with patch("src.cli.run_workflow", return_value=0.42) as run_workflow:
        runner = CliRunner()
        result = runner.invoke(app, ["Will AI?"])
        assert result.exit_code == 0
        assert "0.42" in result.stdout
        run_workflow.assert_called_once_with("Will AI?", False)
