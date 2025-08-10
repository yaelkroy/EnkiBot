import os
import sys
import types

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import asyncio
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

sys.modules.setdefault("pyodbc", types.SimpleNamespace(Connection=object))

from enkibot.core.language_service import LanguageService

# Provide a stub for optional dependencies imported in telegram_handlers
sys.modules.setdefault("tldextract", types.SimpleNamespace())

from enkibot.core import telegram_handlers as th_module

@pytest.mark.parametrize(
    "method_name, expected_text",
    [
        ("_handle_regenerate_reaction", "Пересоздаю ответ..."),
        ("_handle_expand_reaction", "Расширяю предыдущий ответ..."),
        ("_handle_summary_reaction", "Создаю краткое резюме..."),
    ],
)
def test_reaction_handlers_use_reactor_language(method_name, expected_text):
    ls = LanguageService(llm_services=SimpleNamespace(), db_manager=None)
    handler = object.__new__(th_module.TelegramHandlerService)
    handler.language_service = ls

    mock_bot = SimpleNamespace(send_message=AsyncMock())
    context = SimpleNamespace(bot=mock_bot)

    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=42),
        message_reaction=SimpleNamespace(user=SimpleNamespace(language_code="ru"))
    )

    method = getattr(handler, method_name)
    asyncio.run(method(update, context))

    mock_bot.send_message.assert_awaited_once_with(chat_id=42, text=expected_text)
    assert ls.current_lang == "ru"
