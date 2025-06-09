from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from pytest_mock import MockerFixture  # type: ignore

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.tools.fetch_page import fetch_page


def test_fetch_page(mocker: MockerFixture) -> None:
    page = mocker.Mock()
    page.content.return_value = "<html></html>"

    browser = mocker.Mock()
    browser.new_page.return_value = page

    pw = SimpleNamespace(chromium=SimpleNamespace(launch=mocker.Mock(return_value=browser)))
    cm = mocker.MagicMock()
    cm.__enter__.return_value = pw
    cm.__exit__.return_value = False

    mocker.patch("src.tools.fetch_page.sync_playwright", return_value=cm)

    result = fetch_page("http://example.com")

    assert result == "<html></html>"
    page.goto.assert_called_once_with("http://example.com")
    page.content.assert_called_once_with()
    browser.close.assert_called_once_with()
