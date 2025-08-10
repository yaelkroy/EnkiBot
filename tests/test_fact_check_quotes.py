import os
import sys
import types
import pytest
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.modules.setdefault("pyodbc", types.SimpleNamespace(Connection=object))
sys.modules.setdefault("httpx", types.SimpleNamespace())
sys.modules.setdefault("openai", types.SimpleNamespace())


class DummyFilter:
    def __and__(self, other):
        return self

    def __or__(self, other):
        return self

    def __invert__(self):
        return self


dummy_filters = types.SimpleNamespace(
    TEXT=DummyFilter(),
    COMMAND=DummyFilter(),
    CAPTION=DummyFilter(),
)

sys.modules.setdefault(
    "telegram",
    types.SimpleNamespace(
        InlineKeyboardButton=object,
        InlineKeyboardMarkup=object,
        Update=object,
        Message=object,
    ),
)

sys.modules.setdefault(
    "telegram.ext",
    types.SimpleNamespace(
        Application=object,
        CallbackQueryHandler=object,
        CommandHandler=object,
        ContextTypes=types.SimpleNamespace(DEFAULT_TYPE=object),
        MessageHandler=object,
        filters=dummy_filters,
    ),
)

from enkibot.modules.fact_check import QuoteGate, FactChecker


def test_quote_gate_ignores_single_word():
    qg = QuoteGate()
    score = asyncio.run(qg.predict('Мошенники сказали: «Силовики»'))
    assert score < 0.5


def test_extract_quote_ignores_single_word():
    fc = FactChecker()
    assert asyncio.run(fc.extract_quote('«Силовики»')) is None
    assert asyncio.run(fc.extract_quote('«Лиса и ежик шли по лесу»')) is not None
