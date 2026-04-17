from __future__ import annotations

import asyncio
import random
from typing import Any

from ddgs import DDGS
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.services.llm_service import FunctionCallParams

WEB_SEARCH_ACKNOWLEDGEMENTS = (
    "Give me a moment while I look that up.",
    "One moment. Let me check that for you.",
    "Let me look that up for you.",
    "Just a second while I check that.",
)


def _normalize_result(item: dict[str, Any]) -> dict[str, str]:
    return {
        "title": str(item.get("title") or item.get("name") or "").strip(),
        "url": str(item.get("href") or item.get("url") or "").strip(),
        "snippet": str(item.get("body") or item.get("snippet") or "").strip(),
        "source": str(item.get("source") or item.get("publisher") or "").strip(),
    }


async def _speak_web_search_acknowledgement(params: FunctionCallParams) -> None:
    await params.llm.push_frame(
        TTSSpeakFrame(
            text=random.choice(WEB_SEARCH_ACKNOWLEDGEMENTS),
            append_to_context=False,
        )
    )


async def web_search(
    params: FunctionCallParams,
) -> None:
    arguments = dict(params.arguments)
    query = str(arguments.get("query") or "").strip()
    max_results = int(arguments.get("max_results", 5))
    search_type = str(arguments.get("search_type") or "text")
    timelimit_value = arguments.get("timelimit")
    timelimit = None if timelimit_value in (None, "") else str(timelimit_value)

    if not query:
        await params.result_callback({"error": "Search query cannot be empty."})
        return

    bounded_max_results = max(1, min(max_results, 8))
    normalized_search_type = search_type.strip().lower() if search_type else "text"

    await _speak_web_search_acknowledgement(params)

    def run_search() -> list[dict[str, str]]:
        ddgs = DDGS(timeout=10)
        if normalized_search_type == "news":
            raw_results = ddgs.news(
                query,
                max_results=bounded_max_results,
                timelimit=timelimit,
            )
        else:
            raw_results = ddgs.text(
                query,
                max_results=bounded_max_results,
                timelimit=timelimit,
            )
        return [_normalize_result(item) for item in raw_results]

    try:
        results = await asyncio.to_thread(run_search)
    except Exception as exc:
        logger.warning("web_search failed for query={!r}: {}", query, exc)
        await params.result_callback(
            {
                "error": f"Web search failed: {exc}",
                "query": query,
                "search_type": normalized_search_type,
            }
        )
        return

    await params.result_callback(
        {
            "query": query,
            "search_type": normalized_search_type,
            "results_found": len(results),
            "results": results,
        }
    )


WEB_SEARCH_TOOL_SCHEMA = FunctionSchema(
    name="web_search",
    description=(
        "Search the public web for current information. Use this for time-sensitive questions "
        "such as weather, news, current events, prices, product availability, schedules, or "
        "anything phrased as current, latest, today, right now, or recent."
    ),
    properties={
        "query": {
            "type": "string",
            "description": "The web search query to run.",
        },
        "max_results": {
            "type": "integer",
            "description": "How many search results to return. Keep this small, usually 3 to 5.",
        },
        "search_type": {
            "type": "string",
            "description": (
                "Search mode. Use `text` for normal web search or `news` for recent reporting."
            ),
        },
        "timelimit": {
            "type": "string",
            "description": "Optional freshness filter such as `d`, `w`, `m`, or `y`.",
        },
    },
    required=["query"],
)
