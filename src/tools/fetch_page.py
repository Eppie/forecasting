"""Tool for fetching webpage contents using Playwright."""

from __future__ import annotations

from playwright.sync_api import sync_playwright
from pydantic import BaseModel, Field


class FetchArgs(BaseModel):
    """Arguments accepted by the fetch_page tool."""

    url: str = Field(..., description="URL of the page to fetch")


def fetch_page(url: str) -> str:
    """Fetch the HTML content of a webpage.

    Args:
        url: The URL of the page to retrieve.

    Returns:
        The HTML source of the loaded page.
    """
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        content = page.content()
        browser.close()
    return content
