from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from duckduckgo_search import DDGS
from . import Tool, ToolResult

class WebSearchArgs(BaseModel):
    """Arguments for the web search tool."""
    query: str = Field(..., description="Search query string")
    max_results: int = Field(5, description="Maximum number of results to return (1-20)", ge=1, le=20)
    region: str = Field("wt-wt", description="Region for search (e.g., 'us-en', 'uk-en')")
    timelimit: Optional[str] = Field(
        None,
        description="Filter by time: 'd' (day), 'w' (week), 'm' (month), 'y' (year)"
    )

class WebSearchTool(Tool[List[Dict[str, Any]]]):
    """Tool for performing web searches using DuckDuckGo."""
    
    name = "web_search"
    description = "Search the web for information using DuckDuckGo. Useful for finding current information, news, or general knowledge."
    
    def __init__(self):
        self._ddgs = DDGS()
    
    @property
    def args_schema(self) -> type[BaseModel]:
        return WebSearchArgs
    
    def execute(self, **kwargs) -> ToolResult[List[Dict[str, Any]]]:
        try:
            # Validate arguments
            args = WebSearchArgs(**kwargs)
            
            # Perform the search
            results = list(
                self._ddgs.text(
                    args.query,
                    region=args.region,
                    max_results=args.max_results,
                    timelimit=args.timelimit
                )
            )
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append({
                    "title": result.get("title", "No title"),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", ""),
                    "rank": i
                })
            
            return ToolResult(success=True, result=formatted_results)
            
        except Exception as e:
            return ToolResult(success=False, error=f"Search failed: {str(e)}")


class NewsSearchArgs(BaseModel):
    """Arguments for the news search tool."""
    query: str = Field(..., description="Search query string for news")
    max_results: int = Field(5, description="Maximum number of results to return (1-20)", ge=1, le=20)
    region: str = Field("wt-wt", description="Region for search (e.g., 'us-en', 'uk-en')")

class NewsSearchTool(Tool[List[Dict[str, Any]]]):
    """Tool for searching news using DuckDuckGo."""
    
    name = "news_search"
    description = "Search for recent news articles. Useful for finding the latest information on current events."
    
    def __init__(self):
        self._ddgs = DDGS()
    
    @property
    def args_schema(self) -> type[BaseModel]:
        return NewsSearchArgs
    
    def execute(self, **kwargs) -> ToolResult[List[Dict[str, Any]]]:
        try:
            # Validate arguments
            args = NewsSearchArgs(**kwargs)
            
            # Perform the news search
            results = list(
                self._ddgs.news(
                    args.query,
                    region=args.region,
                    max_results=args.max_results
                )
            )
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append({
                    "title": result.get("title", "No title"),
                    "url": result.get("url", ""),
                    "source": result.get("source", ""),
                    "date": result.get("date", ""),
                    "snippet": result.get("body", ""),
                    "rank": i
                })
            
            return ToolResult(success=True, result=formatted_results)
            
        except Exception as e:
            return ToolResult(success=False, error=f"News search failed: {str(e)}")
