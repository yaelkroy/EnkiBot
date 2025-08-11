import os
import sys
import types
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
        ContextTypes=types.SimpleNamespace(DEFAULT_TYPE=object),
        MessageHandler=object,
        filters=filters_stub,
    ),
)

from enkibot.modules.fact_check import FactCheckBot, Claim, Verdict


def test_run_check_uses_positive_reaction_for_true_label():
    fc = SimpleNamespace(
        extract_claim=AsyncMock(return_value=Claim("x", "x", "en", [], "h")),
        research=AsyncMock(
            return_value=Verdict(label="True", confidence=0.9, summary="ok", sources=[])
        ),
    )
    bot = FactCheckBot(app=None, fc=fc)
    target_msg = SimpleNamespace(set_reaction=AsyncMock(), reply_text=AsyncMock())
    update = SimpleNamespace(effective_message=target_msg)
    ctx = SimpleNamespace()

    asyncio.run(bot._run_check(update, ctx, "news text"))

    fc.extract_claim.assert_awaited()
    fc.research.assert_awaited()
    target_msg.set_reaction.assert_awaited_once_with("👍")
    target_msg.reply_text.assert_not_called()

