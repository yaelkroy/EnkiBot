# enkibot/utils/message_utils.py
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
#
# -------------------------------------------------------------------------------
# Future Improvements:
# - Improve modularity to support additional features and services.
# - Enhance error handling and logging for better maintenance.
# - Expand unit tests to cover more edge cases.
# -------------------------------------------------------------------------------

"""Helper utilities for working with Telegram messages."""

from __future__ import annotations

from typing import Any
import re
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode


def is_forwarded_message(message: Any) -> bool:
    """Return ``True`` if the provided Telegram *message* appears to be forwarded.

    This checks several possible attributes that may indicate a forwarded
    message across different versions of the Telegram Bot API. Any missing
    attributes are safely ignored to avoid raising :class:`AttributeError`.
    """
    if message is None:
        return False

    attrs_to_check = (
        "forward_origin",
        "forward_from",
        "forward_from_chat",
        "forward_sender_name",
        "forward_date",
    )
    for attr in attrs_to_check:
        try:
            if getattr(message, attr):
                return True
        except AttributeError:
            continue

    try:
        return bool(getattr(message, "is_automatic_forward"))
    except AttributeError:
        return False


def get_text(message: Any) -> str | None:
    """Return text or caption from a Telegram *message*.

    Many messages such as photos or videos carry their textual content in the
    ``caption`` field rather than ``text``.  This helper consolidates both
    attributes so handlers can process any user supplied text without worrying
    about the underlying message type.
    """

    if message is None:
        return None

    text = getattr(message, "text", None)
    caption = getattr(message, "caption", None)
    return text or caption


URL_PATTERN = re.compile(r"https?://\S+")


def _strip_utm_source(url: str) -> str:
    """Remove ``utm_source=openai`` query parameters from a single *url*."""
    parts = urlsplit(url)
    query_params = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not (k == "utm_source" and v == "openai")]
    new_query = urlencode(query_params)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))


def clean_output_text(text: str | None) -> str | None:
    """Sanitize bot replies by removing tracking parameters and duplicate lines.

    - Strips ``utm_source=openai`` from any URLs in *text*.
    - Removes ``?utm_source=openai`` fragments that appear outside of URLs.
    - Removes consecutive duplicate lines to avoid repeated content.

    Returns ``None`` if the cleaned text would be empty.
    """
    if not text:
        return None

    def repl(match: re.Match[str]) -> str:
        return _strip_utm_source(match.group(0))

    cleaned = URL_PATTERN.sub(repl, text)
    # Also drop tracking fragments left as plain text
    cleaned = re.sub(r"[?&]utm_source=openai", "", cleaned)

    deduped_lines: list[str] = []
    prev_line: str | None = None
    for line in cleaned.splitlines():
        if line != prev_line:
            deduped_lines.append(line)
        prev_line = line

    cleaned_joined = "\n".join(deduped_lines).strip()
    return cleaned_joined or None
