import re
import unicodedata
from typing import Iterable, Tuple

NAME_ALIASES_DEFAULT = [
    r"энки", r"енки", r"энкі", r"енкі", r"enki",
]


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", "")
    return text.strip()


def extract_assistant_prompt(text: str, aliases: Iterable[str], bot_username: str | None = None) -> Tuple[bool, str, str]:
    """Return (triggered, content, alias) from text after bot-name prefix.

    Args:
        text: Original user text.
        aliases: Iterable of alias strings to match.
        bot_username: Telegram username of the bot, if available.
    """
    t = _normalize(text)

    # Normalize and case-fold aliases for robust matching across scripts
    alias_list = {_normalize(a).casefold() for a in aliases if a}
    alias_list.update({_normalize(a).casefold() for a in NAME_ALIASES_DEFAULT})
    patterns = [re.escape(a) for a in alias_list]
    if bot_username:
        bot_username = _normalize(bot_username).casefold()
        patterns.append(re.escape(bot_username))
        patterns.append("@" + re.escape(bot_username))
        if not bot_username.endswith("bot"):
            patterns.append("@" + re.escape(bot_username) + "bot")
    if not patterns:
        return False, "", ""
    alias_group = "|".join(patterns)
    name_re = re.compile(
        rf"^(?:эй[,!:]?\s+|hey[,!:]?\s+)?(?P<alias>{alias_group})[\s,.:;!–—-]*(?P<content>.*)$",
        re.IGNORECASE | re.UNICODE,
    )
    m = name_re.match(t)
    if not m:
        return False, "", ""
    return True, m.group("content").strip(), m.group("alias")
