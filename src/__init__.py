"""Public API for the forecasting package."""

from .ollama_utils import execute_tool_calls, generate_search_queries

__all__ = ["generate_search_queries", "execute_tool_calls"]
