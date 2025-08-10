import os
import sys
import types
import pytest
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.modules.setdefault("pyodbc", types.SimpleNamespace(Connection=object))

from enkibot.core.language_service import LanguageService


class DummyLLM:
    async def call_openai_llm(self, *args, **kwargs):
        return None

    def is_provider_configured(self, *args, **kwargs):
        return False


class DummyDB:
    async def get_recent_chat_texts(self, chat_id, limit):
        return []


def test_russian_heuristic_detection():
    ls = LanguageService(DummyLLM(), DummyDB())
    asyncio.run(ls.determine_language_context('энки расскажи сказку', chat_id=None))
    assert ls.current_lang == 'ru'
