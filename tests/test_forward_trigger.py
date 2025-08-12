import os
import sys
import asyncio
from types import SimpleNamespace

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from enkibot.core.telegram_handlers import TelegramHandlerService


def test_forwarded_message_triggers():
    service = TelegramHandlerService.__new__(TelegramHandlerService)
    service.allowed_group_ids = set()
    service.bot_nicknames = []
    forward_from_chat = SimpleNamespace(id=111, type="channel")
    message = SimpleNamespace(
        chat=SimpleNamespace(id=1, type="supergroup"),
        reply_to_message=None,
        forward_from_chat=forward_from_chat,
        text="Some news",
    )
    update = SimpleNamespace(message=message, effective_chat=message.chat)
    context = SimpleNamespace(bot=SimpleNamespace(username="enkibot"))
    assert asyncio.run(service._is_triggered(update, context, "some news"))
