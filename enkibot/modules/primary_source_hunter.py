# enkibot/modules/primary_source_hunter.py
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
"""Minimal primary source hunter.

This module performs a very small subset of the primary‑source hunting
behaviour described in the project design.  It expands the user query with a
few synonyms and searches official domains first, falling back to wire
services and reputable outlets.  The real system would include more
sophisticated language handling, timestamp extraction and archiving.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Dict

import json
import logging
import openai
from .. import config

logger = logging.getLogger(__name__)


@dataclass
class SourceHit:
    """Lightweight representation of a search hit."""

    url: str
    title: str
    domain: str
    tier: int  # 1=primary, 2=wire, 3=reputable


class PrimarySourceHunter:
    """Searches for primary sources before any summarisation occurs."""

    def __init__(self, max_results: int = 5) -> None:
        self.max_results = max_results
        self.client: openai.AsyncOpenAI | None = None
        if config.OPENAI_API_KEY:
            self.client = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        self.model_id = config.OPENAI_DEEP_RESEARCH_MODEL_ID

        self.official_domains = [
            "gov.uk",
            "eeas.europa.eu",
            "idf.il",
            "mod.gov",
            "mfa.gov",
        ]
        self.wire_domains = ["reuters.com", "apnews.com", "afp.com"]
        self.reputable_domains = [
            "bbc.com",
            "ft.com",
            "nytimes.com",
            "washingtonpost.com",
            "theguardian.com",
        ]

        self.synonyms: Dict[str, List[str]] = {
            "joint statement": ["communiqué", "совместное заявление", "заявление"],
            "line of contact": [
                "line of control",
                "front line",
                "текущая линия соприкосновения",
            ],
            "ceasefire brief": [
                "briefing",
                "заявление пресс-секретаря",
                "брифинг",
            ],
        }

    def expand_queries(self, text: str, langs: List[str]) -> List[str]:
        """Expand the query with simple synonym replacements."""
        queries = [text]
        tl = text.lower()
        for phrase, alts in self.synonyms.items():
            if phrase in tl:
                for alt in alts:
                    queries.append(tl.replace(phrase, alt))
        # In a production system translations would be added here.  We simply
        # deduplicate.
        seen = set()
        expanded: List[str] = []
        for q in queries:
            if q not in seen:
                expanded.append(q)
                seen.add(q)
        return expanded

    async def _web_search(self, query: str) -> List[SourceHit]:
        """Use OpenAI's web search tool to find sources."""
        if not self.client:
            return []
        extra: Dict[str, object] = {}
        if config.OPENAI_SEARCH_CONTEXT_SIZE:
            extra["search_context_size"] = config.OPENAI_SEARCH_CONTEXT_SIZE
        if config.OPENAI_SEARCH_USER_LOCATION:
            try:
                extra["user_location"] = json.loads(
                    config.OPENAI_SEARCH_USER_LOCATION
                )
            except Exception:
                extra["user_location"] = {
                    "country": config.OPENAI_SEARCH_USER_LOCATION
                }
        try:
            resp = await self.client.responses.create(
                model=self.model_id,
                tools=[{"type": "web_search_preview"}],
                tool_choice={"type": "web_search_preview"},
                instructions=(
                    "You are a primary-source hunter. Always include 3-6 sources (at least 1 primary). "
                    "Return a JSON array of objects with 'url' and 'title'."
                ),
                input=query,
                **extra,
            )
            items = json.loads(resp.output_text)
        except Exception as exc:
            logger.warning("Web search failed: %s", exc)
            return []
        hits: List[SourceHit] = []
        for item in items:
            url = item.get("url", "")
            title = item.get("title", "")
            if not url:
                continue
            domain = url.split("/")[2] if "//" in url else url
            tier = 4
            if any(domain.endswith(d) for d in self.official_domains):
                tier = 1
            elif any(domain.endswith(d) for d in self.wire_domains):
                tier = 2
            elif any(domain.endswith(d) for d in self.reputable_domains):
                tier = 3
            hits.append(SourceHit(url=url, title=title, domain=domain, tier=tier))
        return hits

    async def hunt(self, claim_text: str, lang: str | None = None) -> List[SourceHit]:
        """Return a list of source hits prioritising official domains."""
        langs = [lang] if lang else []
        if "en" not in langs:
            langs.append("en")
        queries = self.expand_queries(claim_text, langs)
        hits: List[SourceHit] = []
        for q in queries:
            results = await self._web_search(q)
            for h in results:
                hits.append(h)
                if len(hits) >= self.max_results:
                    return hits
            if hits:
                break
        return hits
