# enkibot/utils/text_splitter.py
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
"""Utility helpers for splitting long text into smaller chunks."""

from __future__ import annotations

from typing import List
import re

MAX_TELEGRAM_MESSAGE_LENGTH = 4096


def split_text_into_chunks(text: str, max_chunk_size: int = MAX_TELEGRAM_MESSAGE_LENGTH) -> List[str]:
    """Split *text* into chunks not exceeding *max_chunk_size* characters.

    The function attempts to respect whitespace boundaries so that words or
    existing newlines are not arbitrarily broken whenever possible.
    """
    if not text:
        return []

    tokens = re.split(r"(\s+)", text)
    chunks: List[str] = []
    current = ""

    for token in tokens:
        if len(current) + len(token) <= max_chunk_size:
            current += token
        else:
            if current:
                chunks.append(current.rstrip())
            current = token.lstrip()
            # If a single token itself exceeds max_chunk_size, hard split it
            while len(current) > max_chunk_size:
                chunks.append(current[:max_chunk_size])
                current = current[max_chunk_size:]

    if current:
        chunks.append(current.rstrip())

    return [c for c in chunks if c]
