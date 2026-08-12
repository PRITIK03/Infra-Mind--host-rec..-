"""
Tavily web-search tool wrapper.

Provides a small factory that returns a LangChain-compatible Tavily
search tool configured from environment settings. Search is optional:
callers should treat WebSearchUnavailableError as a soft skip.
"""

from __future__ import annotations

import os
from typing import Any

from app.config import ConfigError, get_tavily_settings


class WebSearchUnavailableError(RuntimeError):
    """Raised when Tavily search cannot be configured."""


def get_search_tool(max_results: int = 3) -> Any:
    """Returns a TavilySearch tool configured with this project's settings."""
    try:
        settings = get_tavily_settings()
    except ConfigError as exc:
        raise WebSearchUnavailableError(
            "Missing Tavily configuration: set TAVILY_API_KEY in the environment."
        ) from exc

    try:
        from langchain_tavily import TavilySearch
    except ImportError as exc:
        raise WebSearchUnavailableError(
            "Tavily search dependency is missing. Install requirements to enable web search."
        ) from exc

    # TavilySearch reads from env by default; set it explicitly to avoid
    # relying on shell-level export state.
    os.environ["TAVILY_API_KEY"] = settings.api_key
    try:
        return TavilySearch(max_results=max_results)
    except Exception as exc:
        raise WebSearchUnavailableError(
            f"Failed to initialize Tavily search tool: {exc}"
        ) from exc


def try_get_search_tool(max_results: int = 3) -> Any | None:
    """Like get_search_tool, but returns None when search is not configured."""
    try:
        return get_search_tool(max_results=max_results)
    except WebSearchUnavailableError:
        return None
