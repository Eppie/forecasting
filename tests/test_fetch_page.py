from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

playwright_module = ModuleType("playwright")
sync_api_module = ModuleType("playwright.sync_api")
sync_api_module.sync_playwright = MagicMock()  # type: ignore[attr-defined]
playwright_module.sync_api = sync_api_module  # type: ignore[attr-defined]
sys.modules.setdefault("playwright", playwright_module)
sys.modules.setdefault("playwright.sync_api", sync_api_module)

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.tools.fetch_page import fetch_page  # noqa: E402


def test_fetch_page() -> None:
    page = MagicMock()
    page.content.return_value = "<html></html>"

    browser = MagicMock()
    browser.new_page.return_value = page

    pw = SimpleNamespace(chromium=SimpleNamespace(launch=MagicMock(return_value=browser)))
    cm = MagicMock()
    cm.__enter__.return_value = pw
    cm.__exit__.return_value = False

    with patch("src.tools.fetch_page.sync_playwright", return_value=cm):
        result = fetch_page("http://example.com")

    assert result == "<html></html>"
    page.goto.assert_called_once_with("http://example.com")
    page.content.assert_called_once_with()
    browser.close.assert_called_once_with()
