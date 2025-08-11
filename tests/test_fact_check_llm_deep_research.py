import asyncio
import json
import os
import sys
import types

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
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

from enkibot.modules.fact_check import FactChecker, Claim


class DummyLLM:
    def __init__(self):
        self.deepr_called = False
        self.llm_called = False
        self.messages = None

    async def call_openai_deep_research(self, messages, max_output_tokens=1000):
        self.deepr_called = True
        self.messages = messages
        return json.dumps({
            "label": "unverified",
            "confidence": 0.0,
            "summary": "пример",
        })

    async def call_openai_llm(self, *args, **kwargs):
        self.llm_called = True
        return None


def test_llm_verdict_uses_deep_research_and_russian_summary():
    dummy = DummyLLM()
    fc = FactChecker(llm_services=dummy)
    claim = Claim(
        text_norm="Умер театральный режиссер Юрий Бутусов.",
        text_orig="Умер театральный режиссер Юрий Бутусов.",
        lang="ru",
        urls=[],
        hash="abc",
    )
    verdict = asyncio.run(fc._llm_verdict(claim, []))
    assert dummy.deepr_called
    assert not dummy.llm_called
    assert "summary' in Russian" in dummy.messages[0]["content"]
    assert verdict.summary == "пример"
