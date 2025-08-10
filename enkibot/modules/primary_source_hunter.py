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

from duckduckgo_search import DDGS


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
        self.ddgs = DDGS()

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

    def _search_domain(self, query: str, domain: str) -> Iterable[SourceHit]:
        """Search DuckDuckGo for a given domain."""
        search_q = f"{query} site:{domain}"
        for hit in self.ddgs.text(search_q, max_results=1):
            yield SourceHit(
                url=hit.get("href", ""),
                title=hit.get("title", ""),
                domain=domain,
                tier=0,
            )

    async def hunt(self, claim_text: str, lang: str | None = None) -> List[SourceHit]:
        """Return a list of source hits prioritising official domains."""
        langs = [lang] if lang else []
        if "en" not in langs:
            langs.append("en")
        queries = self.expand_queries(claim_text, langs)
        hits: List[SourceHit] = []

        # Stage A: primary
        for q in queries:
            for domain in self.official_domains:
                for h in self._search_domain(q, domain):
                    h.tier = 1
                    hits.append(h)
                    if len(hits) >= self.max_results:
                        return hits
            if hits:
                return hits

        # Stage B: wire services
        for q in queries:
            for domain in self.wire_domains:
                for h in self._search_domain(q, domain):
                    h.tier = 2
                    hits.append(h)
                    if len(hits) >= self.max_results:
                        return hits
            if hits:
                return hits

        # Stage C: reputable outlets
        for q in queries:
            for domain in self.reputable_domains:
                for h in self._search_domain(q, domain):
                    h.tier = 3
                    hits.append(h)
                    if len(hits) >= self.max_results:
                        return hits

        return hits
