# enkibot/modules/web_tool.py
# EnkiBot: Advanced Multilingual Telegram AI Assistant
# Copyright (C) 2025 Yael Demedetskaya <yaelkroy@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# -------------------------------------------------------------------------------
# Future Improvements:
# - Improve modularity to support additional features and services.
# - Enhance error handling and logging for better maintenance.
# - Expand unit tests to cover more edge cases.
# -------------------------------------------------------------------------------
"""Light‑weight web research helper used by local models."""
from __future__ import annotations

from typing import List, Dict
import logging
import json

import openai
import requests
import trafilatura
from .. import config

logger = logging.getLogger(__name__)


def web_research(query: str, k: int = 5) -> List[Dict[str, str]]:
    """Return a list of documents for the given search query.

    Each document is a mapping containing ``title``, ``url`` and ``text`` keys.
    ``text`` is truncated to a reasonable length for prompt‑embedding.
    """

    logger.info("Web research for query: %s", query)
    if not config.OPENAI_API_KEY:
        return []
    client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
    extra: Dict[str, object] = {}
    if config.OPENAI_SEARCH_CONTEXT_SIZE:
        extra["search_context_size"] = config.OPENAI_SEARCH_CONTEXT_SIZE
    if config.OPENAI_SEARCH_USER_LOCATION:
        try:
            extra["user_location"] = json.loads(
                config.OPENAI_SEARCH_USER_LOCATION
            )
        except Exception:
            extra["user_location"] = {"country": config.OPENAI_SEARCH_USER_LOCATION}
    try:
        resp = client.responses.create(
            model=config.OPENAI_DEEP_RESEARCH_MODEL_ID,
            tools=[{"type": "web_search_preview"}],
            tool_choice={"type": "web_search_preview"},
            instructions=(
                f"Return up to {k} sources as a JSON array of objects with 'title' and 'url'."
            ),
            input=query,
            **extra,
        )
        hits = json.loads(resp.output_text)
    except Exception as exc:  # pragma: no cover
        logger.warning("Web search failed: %s", exc)
        return []
    docs: List[Dict[str, str]] = []
    for hit in hits[:k]:
        url = hit.get("url")
        if not url:
            continue
        try:
            html = requests.get(url, timeout=15).text
            text = trafilatura.extract(html) or ""
            if text:
                docs.append({
                    "title": hit.get("title", ""),
                    "url": url,
                    "text": text[:20000],
                })
        except Exception as exc:  # pragma: no cover - network failures common
            logger.warning("Failed to fetch %s: %s", url, exc)
    return docs
