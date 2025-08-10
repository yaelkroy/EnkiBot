#!/usr/bin/env python3
"""Utility to check localization coverage across language packs.

This script compares the key structure of all JSON files in ``enkibot/lang``
and reports any missing or extra keys for each locale.  Optionally it can
produce a pseudo‑locale ``x-ps.json`` which wraps each string in brackets and
pads it.  The pseudo locale is useful for spotting untranslated or truncated
strings during manual QA.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Set


LANG_DIR = Path(__file__).resolve().parents[1] / "enkibot" / "lang"


def _load_locale(path: Path) -> Dict:
    with path.open("r", encoding="utf-8-sig") as f:  # handle BOM if present
        return json.load(f)


def _flatten(d: Dict, prefix: str = "") -> Iterable[str]:
    for key, value in d.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from _flatten(value, new_key)
        else:
            yield new_key


def _compare(locales: Dict[str, Dict]) -> None:
    key_sets: Dict[str, Set[str]] = {
        lang: set(_flatten(data)) for lang, data in locales.items()
    }
    base_lang = next(iter(key_sets))
    base_keys = key_sets[base_lang]
    ok = True
    for lang, keys in key_sets.items():
        missing = base_keys - keys
        extra = keys - base_keys
        if missing or extra:
            ok = False
            if missing:
                print(f"[{lang}] missing keys compared to {base_lang}:")
                for k in sorted(missing):
                    print("  ", k)
            if extra:
                print(f"[{lang}] extra keys compared to {base_lang}:")
                for k in sorted(extra):
                    print("  ", k)
    if ok:
        print("All locale files share the same key set.")


def _pseudo_localize(base: Dict) -> Dict:
    def transform(obj):
        if isinstance(obj, dict):
            return {k: transform(v) for k, v in obj.items()}
        if isinstance(obj, str):
            padded = obj
            if len(padded) < 10:
                padded += "*" * (10 - len(padded))
            return f"[ {padded} ]"
        return obj

    return transform(base)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check i18n coverage and optionally generate pseudo locale"
    )
    parser.add_argument(
        "--pseudo",
        action="store_true",
        help="Generate x-ps.json pseudo locale from en.json",
    )
    args = parser.parse_args()

    locales = {p.stem: _load_locale(p) for p in LANG_DIR.glob("*.json")}
    if not locales:
        print("No locale files found in", LANG_DIR, file=sys.stderr)
        sys.exit(1)

    _compare(locales)

    if args.pseudo:
        base = locales.get("en")
        if base is None:
            print("en.json not found; cannot generate pseudo locale", file=sys.stderr)
            sys.exit(1)
        pseudo_path = LANG_DIR / "x-ps.json"
        pseudo = _pseudo_localize(base)
        with pseudo_path.open("w", encoding="utf-8") as f:
            json.dump(pseudo, f, ensure_ascii=False, indent=2)
        print(f"Pseudo locale written to {pseudo_path}")


if __name__ == "__main__":
    main()
