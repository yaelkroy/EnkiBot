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


def test_ukrainian_detection():
    ls = LanguageService(DummyLLM(), DummyDB())
    asyncio.run(ls.determine_language_context('привіт, як справи?', chat_id=None))
    assert ls.current_lang == 'uk'


def test_english_detection():
    ls = LanguageService(DummyLLM(), DummyDB())
    asyncio.run(ls.determine_language_context('hello there', chat_id=None))
    assert ls.current_lang == 'en'


def test_code_switch_prefers_cyrillic():
    ls = LanguageService(DummyLLM(), DummyDB())
    asyncio.run(ls.determine_language_context('привет show weather', chat_id=None))
    assert ls.current_lang == 'ru'


def test_transliteration_detection():
    ls = LanguageService(DummyLLM(), DummyDB())
    asyncio.run(ls.determine_language_context('privet, kak dela?', chat_id=None))
    assert ls.current_lang == 'ru'


def test_short_inputs():
    ls = LanguageService(DummyLLM(), DummyDB())
    asyncio.run(ls.determine_language_context('да', chat_id=None))
    assert ls.current_lang == 'ru'
    asyncio.run(ls.determine_language_context('ok', chat_id=None))
    assert ls.current_lang == 'en'


class DummyDBHistory:
    async def get_recent_chat_texts(self, chat_id, limit):
        return ['Новость про Київ']


def test_english_reply_to_russian_history():
    ls = LanguageService(DummyLLM(), DummyDBHistory())
    asyncio.run(ls.determine_language_context('Where is the proof?', chat_id=1))
    assert ls.current_lang == 'en'
