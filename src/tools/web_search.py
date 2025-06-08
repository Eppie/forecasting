import os, asyncio
from typing import List, Dict
from pydantic import BaseModel, Field
from brave_search import BraveSearch, WebSearchRequest

class BraveArgs(BaseModel):
    """Arguments accepted by the brave_search tool."""
    query: str = Field(..., description="Plain-English search query")
    limit: int = Field(10, ge=1, le=25, description="Max # of hits")

def brave_search(query: str, limit: int = 10) -> List[Dict[str, str]]:
    """
    Search the public web with Brave and return compact JSON hits.
    Designed for Ollama's function-calling interface.
    """
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        raise RuntimeError("Set BRAVE_SEARCH_API_KEY")

    client = BraveSearch(api_key=api_key)

    async def _go() -> List[Dict[str, str]]:
        resp = await client.web(WebSearchRequest(q=query, count=limit))
        hits = resp.web.results if resp.web else []
        return [
            {"title": h.title, "url": h.url, "snippet": h.description}
            for h in hits
        ]

    return asyncio.run(_go())
