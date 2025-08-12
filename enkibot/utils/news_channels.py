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
    """Fetch the TLGRM news channel directory and return usernames."""
    logger.info("Requesting news channel directory from %s", NEWS_CHANNELS_URL)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(NEWS_CHANNELS_URL)
            logger.info(
                "Received response %s with %d bytes",
                resp.status_code,
                len(resp.text),
            )
            resp.raise_for_status()
        usernames = extract_channel_usernames(resp.text)
        logger.info("Fetched %d total usernames", len(usernames))
        return usernames
    except Exception as exc:  # pragma: no cover - network errors
        logger.error("Failed to fetch news channels: %s", exc)
        return []
