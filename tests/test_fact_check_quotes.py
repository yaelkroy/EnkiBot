import os
import sys
import types
import pytest
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.modules.setdefault("pyodbc", types.SimpleNamespace(Connection=object))

from enkibot.modules.fact_check import QuoteGate, FactChecker


def test_quote_gate_ignores_single_word():
    qg = QuoteGate()
    score = asyncio.run(qg.predict('Мошенники сказали: «Силовики»'))
    assert score < 0.5


def test_extract_quote_ignores_single_word():
    fc = FactChecker()
    assert asyncio.run(fc.extract_quote('«Силовики»')) is None
    assert asyncio.run(fc.extract_quote('«Лиса и ежик шли по лесу»')) is not None
