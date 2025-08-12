import os
import sys
import types

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.modules.setdefault("pyodbc", types.SimpleNamespace(Connection=object))

from enkibot.utils.message_utils import clean_output_text


def test_clean_output_text_removes_utm_from_url():
    text = "Check https://example.com/path?utm_source=openai&x=1"
    expected = "Check https://example.com/path?x=1"
    assert clean_output_text(text) == expected


def test_clean_output_text_removes_utm_from_plain_text():
    text = "something?utm_source=openai else"
    assert clean_output_text(text) == "something else"


def test_clean_output_text_returns_none_when_empty_after_cleaning():
    assert clean_output_text("?utm_source=openai") is None
