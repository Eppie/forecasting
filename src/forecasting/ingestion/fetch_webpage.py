from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from charset_normalizer import from_bytes

from playwright.async_api import async_playwright, Playwright, Page


class FetchError(RuntimeError):
    """Raised when fetch_page ultimately fails."""


def _decode(body: bytes, headers: Dict[str, str]) -> str:
    enc = headers.get("content-type", "")
    if "charset=" in enc:
        charset = enc.split("charset=")[-1].split(";")[0].strip()
        try:
            return body.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):
            pass
    # If declared charset isn’t usable, explicitly decode as UTF-8 fallback:
    try:
        return body.decode("utf-8", errors="replace")
    except UnicodeDecodeError:
        return from_bytes(body).best().output()


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((httpx.HTTPError, FetchError)),
)
async def _fetch_static(
        url: str,
        *,
        timeout: float,
        headers: Optional[Dict[str, str]] = None,
        **client_kwargs: Any,
) -> str:
    async with httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
            follow_redirects=True,
            http2=True,
            **client_kwargs,
    ) as client:
        resp: httpx.Response = await client.get(url)
        resp.raise_for_status()
        return _decode(resp.content, resp.headers)


async def _fetch_js(
        url: str,
        *,
        timeout: float,
        headers: Optional[Dict[str, str]] = None,
        wait_until: str = "networkidle",
) -> str:
    if async_playwright is None:
        raise FetchError("Playwright not installed; cannot render JS")

    async with async_playwright() as pw:  # type: Playwright
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(extra_http_headers=headers or {})
        page: Page = await context.new_page()
        await page.goto(url, timeout=int(timeout * 1000), wait_until=wait_until)
        content: str = await page.content()
        await browser.close()
        return content


async def _fetch_page_async(
        url: str,
        *,
        render_js: bool = False,
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
) -> str:
    if render_js:
        return await _fetch_js(url, timeout=timeout, headers=headers, **kwargs)
    return await _fetch_static(url, timeout=timeout, headers=headers, **kwargs)


def fetch_page(
        url: str,
        *,
        render_js: bool = False,
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
) -> str:
    """
    Retrieve the fully-rendered HTML for *url*.

    Args:
        url: Target URL.
        render_js: If True, use Playwright to execute JavaScript.
        timeout: Request deadline in seconds.
        headers: Additional request headers; overrides defaults.
        **kwargs: Forwarded to the underlying httpx.AsyncClient or Playwright.

    Returns:
        Raw HTML string.

    Raises:
        FetchError | httpx.HTTPError
    """
    try:
        return asyncio.run(_fetch_page_async(url,
                                             render_js=render_js,
                                             timeout=timeout,
                                             headers=headers,
                                             **kwargs))
    except Exception as exc:  # pragma: no cover
        raise FetchError(f"Failed to fetch {url}: {exc}") from exc


if __name__ == "__main__":
    """
    Demonstration of fetch_page for:
      1. An easy/static URL (no JS required).
      2. A hard/dynamic URL (JS rendering required).
    """

    # 1) Easy URL: static html (no client-side JS needed)
    easy_url = "https://example.com"
    try:
        print(f"Fetching static content from {easy_url!r} ...")
        static_html = fetch_page(easy_url, render_js=False, timeout=10.0)
        print(f"Fetched {len(static_html)} bytes from {easy_url!r}.\n")
        print("First 200 characters of the page:")
        print(static_html[:200].replace("\n", " "))
        print("\n" + "-" * 80 + "\n")
    except Exception as e:
        print(f"Error fetching {easy_url!r}: {e}\n")

    # 2) Hard URL: requires JavaScript to render content
    #    Here we pick a known demo site that loads its data via JS.
    #    (Replace with any JS-heavy page you want to test.)
    hard_url = "https://quotes.toscrape.com/js/"
    try:
        print(f"Fetching JS-rendered content from {hard_url!r} ...")
        js_html = fetch_page(hard_url, render_js=True, timeout=30.0)
        print(f"Fetched {len(js_html)} bytes from {hard_url!r}.\n")
        print("First 200 characters of the rendered page:")
        print(js_html.replace("\n", " "))
        print("\n" + "-" * 80 + "\n")
    except Exception as e:
        print(f"Error fetching {hard_url!r}: {e}\n")
