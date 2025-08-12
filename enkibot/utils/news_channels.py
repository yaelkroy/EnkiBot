# enkibot/utils/news_channels.py
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
"""Utilities for retrieving and parsing Telegram news channel usernames."""

from __future__ import annotations

import logging
import re
from typing import List

import httpx

logger = logging.getLogger(__name__)

NEWS_CHANNELS_URL = "https://tlgrm.ru/channels/news"
# Additional category to scrape alongside the default news page.  The exact
# category is not critical for tests as network calls are mocked, but in
# production this points to a second TLGRM catalogue page.
TECH_CHANNELS_URL = "https://tlgrm.ru/channels/technology"

# List of TLGRM catalogue pages that should be scraped.  The fetch helper will
# iterate over all URLs in this sequence.
CHANNEL_CATEGORY_URLS = [NEWS_CHANNELS_URL, TECH_CHANNELS_URL]

# Only pick channel usernames from links to tlgrm or tg://resolve to avoid
# matching image resolution hints like ``@2x`` that appear in the page markup.
_CHANNEL_PATTERN = re.compile(
    r'href=["\'](?:https://tlgrm\.ru/channels/@|tg://resolve\?domain=)([A-Za-z0-9_]+)["\']'
)


def extract_channel_usernames(html: str) -> List[str]:
    """Extract unique channel usernames from *html* content.

    Returned usernames do not include the leading '@'.
    """
    usernames = sorted(set(_CHANNEL_PATTERN.findall(html)))
    logger.info("Extracted %d channel usernames from HTML", len(usernames))
    return usernames


async def fetch_channel_usernames() -> List[str]:
    """Fetch TLGRM channel directories and return unique usernames.

    Each page listed in :data:`CHANNEL_CATEGORY_URLS` is retrieved.  The TLGRM
    catalogue uses endless scrolling and returns only the first page of
    channels on the initial request.  Additional pages are accessible via the
    ``?page=N`` query parameter.  This helper follows the pagination links for
    every configured page so that callers receive the full list of available
    channels.
    """

    logger.info(
        "Requesting channel directories from %s", ", ".join(CHANNEL_CATEGORY_URLS)
    )
    try:
        usernames: set[str] = set()
        total_pages = 0
        async with httpx.AsyncClient(timeout=30.0) as client:
            for base_url in CHANNEL_CATEGORY_URLS:
                page = 1
                last_page = None
                while True:
                    url = base_url if page == 1 else f"{base_url}?page={page}"
                    resp = await client.get(url)
                    logger.info(
                        "Received response %s with %d bytes for %s page %d",
                        resp.status_code,
                        len(resp.text),
                        base_url,
                        page,
                    )
                    resp.raise_for_status()

                    usernames.update(extract_channel_usernames(resp.text))

                    if last_page is None:
                        match = re.search(r'data-last-page="(\d+)"', resp.text)
                        last_page = int(match.group(1)) if match else 1
                        logger.info("Detected %d pages in total for %s", last_page, base_url)

                    if page >= last_page:
                        break
                    page += 1
                total_pages += page

        names = sorted(usernames)
        logger.info(
            "Fetched %d total usernames from %d pages across %d directories",
            len(names),
            total_pages,
            len(CHANNEL_CATEGORY_URLS),
        )
        return names
    except Exception as exc:  # pragma: no cover - network errors
        logger.error("Failed to fetch channel directories: %s", exc)
        return []
