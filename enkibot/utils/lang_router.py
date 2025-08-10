import re
import unicodedata


def normalize(s: str) -> str:
    """Normalize unicode string and drop zero-width characters.

    This utility applies NFKC normalization and removes common zero-width
    characters that may cause message truncation or other subtle bugs.
    """
    s = unicodedata.normalize("NFKC", s or "")
    return re.sub(r"[\u200B\u200C\u200D\u200E\u2060]", "", s).strip()
