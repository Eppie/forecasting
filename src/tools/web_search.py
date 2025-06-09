import asyncio
import os

from brave_search_python_client import BraveSearch, WebSearchRequest  # type: ignore
from pydantic import BaseModel, Field


class SearchArgs(BaseModel):
    """Arguments accepted by the web_search tool."""

    query: str = Field(..., description="Plain-English search query")
    limit: int = Field(10, ge=1, le=10, description="Max # of hits")


def web_search(query: str, limit: int = 10) -> list[dict[str, str]]:
    """
    Search the public web and return compact JSON hits.
    Designed for Ollama's function-calling interface.
    """
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        raise RuntimeError("Set BRAVE_SEARCH_API_KEY")

    client = BraveSearch(api_key=api_key)

    async def _go() -> list[dict[str, str]]:
        resp = await client.web(WebSearchRequest(q=query, count=limit))
        hits = resp.web.results if resp.web else []
        return [{"title": h.title, "url": h.url, "snippet": h.description} for h in hits]

    return asyncio.run(_go())
