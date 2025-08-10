import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from enkibot.utils.trigger_extractor import extract_assistant_prompt

ALIASES = ["энки", "енки", "enki"]


def test_basic_russian():
    triggered, content, alias = extract_assistant_prompt("энки расскажи сказку", ALIASES)
    assert triggered
    assert content == "расскажи сказку"
    assert alias.lower() == "энки"


def test_alias_variant():
    triggered, content, alias = extract_assistant_prompt("енки, составь список", ALIASES)
    assert triggered
    assert content == "составь список"


def test_mixed_punctuation():
    triggered, content, alias = extract_assistant_prompt("Энки—подскажи, как дела?", ALIASES)
    assert triggered
    assert "подскажи" in content.lower()


def test_no_trigger():
    triggered, content, alias = extract_assistant_prompt("расскажи сказку", ALIASES)
    assert not triggered
    assert content == ""


def test_latin_name():
    triggered, content, alias = extract_assistant_prompt("Enki, write a poem", ALIASES)
    assert triggered
    assert content == "write a poem"


def test_alias_with_zero_width_space_in_config():
    custom_aliases = ["бот\u200b"]
    triggered, content, alias = extract_assistant_prompt("бот расскажи сказку", custom_aliases)
    assert triggered
    assert content == "расскажи сказку"
