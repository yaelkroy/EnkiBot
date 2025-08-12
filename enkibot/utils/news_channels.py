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
_CHANNEL_PATTERN = re.compile(r"@([A-Za-z0-9_]+)")


def extract_channel_usernames(html: str) -> List[str]:
    """Extract unique channel usernames from *html* content.

    Returned usernames do not include the leading '@'.
    """
    return sorted(set(_CHANNEL_PATTERN.findall(html)))


async def fetch_channel_usernames() -> List[str]:
    """Fetch the TLGRM news channel directory and return usernames."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(NEWS_CHANNELS_URL)
            resp.raise_for_status()
        return extract_channel_usernames(resp.text)
    except Exception as exc:  # pragma: no cover - network errors
        logger.error("Failed to fetch news channels: %s", exc)
        return []
