import os
import sys
import types
import pytest
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.modules.setdefault("pyodbc", types.SimpleNamespace(Connection=object))
sys.modules.setdefault("telegram", types.SimpleNamespace(Update=object))
sys.modules.setdefault("httpx", types.SimpleNamespace())
sys.modules.setdefault("openai", types.SimpleNamespace())

from enkibot.core.language_service import LanguageService


class DummyLLM:
    async def call_openai_llm(self, *args, **kwargs):
        return None

    def is_provider_configured(self, *args, **kwargs):
        return False


class DummyDB:
    async def get_recent_chat_texts(self, chat_id, limit):
        return []



@pytest.mark.parametrize(
    "text, expected",
    [
        ("энки расскажи сказку", "ru"),
        ("tell me a story", "en"),
        ("енкі розкажи казку", "uk"),
        ("privet, kak dela?", "ru"),
        ("ok", "en"),
        ("ок", "ru"),
        ("да", "ru"),
    ],
)
def test_language_detection_cases(text, expected):
    ls = LanguageService(DummyLLM(), DummyDB())
    asyncio.run(ls.determine_language_context(text, chat_id=None))
    assert ls.current_lang == expected
