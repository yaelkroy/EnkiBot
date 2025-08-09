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

from duckduckgo_search import DDGS
import requests
import trafilatura

logger = logging.getLogger(__name__)


def web_research(query: str, k: int = 5) -> List[Dict[str, str]]:
    """Return a list of documents for the given search query.

    Each document is a mapping containing ``title``, ``url`` and ``text`` keys.
    ``text`` is truncated to a reasonable length for prompt‑embedding.
    """

    logger.info("Web research for query: %s", query)
    hits = list(DDGS().text(query, max_results=k))
    docs: List[Dict[str, str]] = []
    for hit in hits:
        try:
            html = requests.get(hit["href"], timeout=15).text
            text = trafilatura.extract(html) or ""
            if text:
                docs.append({
                    "title": hit.get("title", ""),
                    "url": hit["href"],
                    "text": text[:20000],
                })
        except Exception as exc:  # pragma: no cover - network failures common
            logger.warning("Failed to fetch %s: %s", hit.get("href"), exc)
    return docs
