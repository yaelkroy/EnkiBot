import asyncio
import os
import sys
import types

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.modules.setdefault("pyodbc", types.SimpleNamespace(Connection=object))
filters_stub = types.SimpleNamespace(TEXT=1, COMMAND=2, CAPTION=4)
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
        ContextTypes=object,
        MessageHandler=object,
        filters=filters_stub,
    ),
)

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
