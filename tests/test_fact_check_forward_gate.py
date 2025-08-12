import os
import sys
import types
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.modules.setdefault("pyodbc", types.SimpleNamespace(Connection=object))

from enkibot.modules.fact_check import FactCheckBot


def build_bot(db_usernames):
    app = SimpleNamespace(add_handler=lambda *a, **k: None)
    fc = SimpleNamespace()
    db_manager = SimpleNamespace(
        get_news_channel_usernames=AsyncMock(return_value=db_usernames),
        log_fact_gate=AsyncMock(),
    )
    cfg = lambda _id: {"satire": {"enabled": False}, "auto": {"auto_check_news": True}}
    bot = FactCheckBot(app=app, fc=fc, cfg_reader=cfg, db_manager=db_manager, language_service=SimpleNamespace(get_response_string=lambda *a, **k: ""))
    bot._run_check = AsyncMock()
    return bot


def make_update(username, text="sample"):
    msg = SimpleNamespace(
        text=text,
        message_id=1,
        forward_from_chat=SimpleNamespace(username=username) if username else None,
        photo=None,
        video=None,
        document=None,
        reply_text=AsyncMock(),
        set_reaction=AsyncMock(),
    )
    update = SimpleNamespace(effective_message=msg, effective_chat=SimpleNamespace(id=123))
    return update


def test_forward_from_known_channel_triggers_news_check():
    bot = build_bot({"known"})
    update = make_update("known")
    ctx = SimpleNamespace()
    asyncio.run(bot.on_forward(update, ctx))
    bot._run_check.assert_awaited_once()
    args, kwargs = bot._run_check.await_args
    assert kwargs.get("track") == "news"


def test_forward_unknown_channel_news_gate_triggers_check(monkeypatch):
    bot = build_bot(set())
    bot.news_gate.predict = AsyncMock(return_value=0.8)
    bot.quote_gate.predict = AsyncMock(return_value=0.1)
    update = make_update("unknown")
    ctx = SimpleNamespace()
    asyncio.run(bot.on_forward(update, ctx))
    bot._run_check.assert_awaited_once()
    args, kwargs = bot._run_check.await_args
    assert kwargs.get("track") == "news"


def test_forward_unknown_channel_quote_gate_triggers_check(monkeypatch):
    bot = build_bot(set())
    bot.news_gate.predict = AsyncMock(return_value=0.1)
    bot.quote_gate.predict = AsyncMock(return_value=0.9)
    update = make_update("unknown")
    ctx = SimpleNamespace()
    asyncio.run(bot.on_forward(update, ctx))
    bot._run_check.assert_awaited_once()
    args, kwargs = bot._run_check.await_args
    assert kwargs.get("track") == "book"


def test_forward_without_channel_ignored():
    bot = build_bot(set())
    update = make_update(None)
    ctx = SimpleNamespace()
    asyncio.run(bot.on_forward(update, ctx))
    bot._run_check.assert_not_called()
