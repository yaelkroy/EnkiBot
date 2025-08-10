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


def test_english_detection():
    ls = LanguageService(DummyLLM(), DummyDB())
    asyncio.run(ls.determine_language_context('hello how are you', chat_id=None))
    assert ls.current_lang == 'en'


def test_ukrainian_detection():
    ls = LanguageService(DummyLLM(), DummyDB())
    asyncio.run(ls.determine_language_context('привіт, як справи?', chat_id=None))
    assert ls.current_lang == 'uk'


def test_transliterated_russian_detection():
    ls = LanguageService(DummyLLM(), DummyDB())
    asyncio.run(ls.determine_language_context('privet, kak dela?', chat_id=None))
    assert ls.current_lang == 'ru'


def test_latest_message_priority_over_context():
    class ContextDB(DummyDB):
        async def get_recent_chat_texts(self, chat_id, limit):
            return ['привет из прошлого']

    ls = LanguageService(DummyLLM(), ContextDB())
    asyncio.run(ls.determine_language_context('ok thanks', chat_id=1))
    assert ls.current_lang == 'en'


def test_short_inputs():
    ls = LanguageService(DummyLLM(), DummyDB())
    asyncio.run(ls.determine_language_context('да', chat_id=None))
    assert ls.current_lang == 'ru'
    asyncio.run(ls.determine_language_context('ok', chat_id=None))
    assert ls.current_lang == 'en'
    asyncio.run(ls.determine_language_context('ок', chat_id=None))
    assert ls.current_lang == 'ru'
