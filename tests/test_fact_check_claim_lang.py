import asyncio
import os
import sys
import types

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.modules.setdefault("pyodbc", types.SimpleNamespace(Connection=object))

from enkibot.modules.fact_check import FactChecker


def test_extract_claim_detects_russian():
    text = (
        "Умер театральный режиссер Юрий Бутусов. Ему было 63 года.\n\n"
        "Бутусов был главным режиссёром Театра имени Евгения Вахтангова с 2018 года по 2022 год"
    )
    fc = FactChecker()
    claim = asyncio.run(fc.extract_claim(text))
    assert claim is not None
    assert claim.lang == "ru"
