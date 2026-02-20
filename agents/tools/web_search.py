"""
Web Search Tool — Internet search via DuckDuckGo.
"""

import logging
from typing import List, Dict

from agents.tools.registry import registry, RiskLevel

logger = logging.getLogger(__name__)


@registry.register(
    name="web_search",
    description="Search the internet for information. Returns top results with titles, URLs, and snippets.",
    risk_level=RiskLevel.MEDIUM,
    parameters={"query": "Search query string", "max_results": "Maximum results (default 5)"},
)
def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search the internet using DuckDuckGo."""
    try:
        import httpx

        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }

        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, params=params)
            data = resp.json()

        results = []

        # Abstract (main answer)
        if data.get("Abstract"):
            results.append({
                "title": data.get("Heading", ""),
                "url": data.get("AbstractURL", ""),
                "snippet": data["Abstract"],
            })

        # Related topics
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title": topic.get("Text", "")[:80],
                    "url": topic.get("FirstURL", ""),
                    "snippet": topic.get("Text", ""),
                })

        return results[:max_results]

    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return [{"title": "Error", "url": "", "snippet": f"Search failed: {e}"}]
