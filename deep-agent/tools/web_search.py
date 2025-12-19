"""
Web search tool using Tavily API.

Provides web search capabilities for research and information gathering.
"""

import os
from typing import Literal
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def web_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general"
) -> dict:
    """
    Search the web for information.

    Args:
        query: Search query string
        max_results: Max results to return (default 5)
        topic: 'general', 'news', or 'finance'

    Returns:
        Search results with URLs, content, and metadata
    """
    return tavily_client.search(query, max_results=max_results, topic=topic)
