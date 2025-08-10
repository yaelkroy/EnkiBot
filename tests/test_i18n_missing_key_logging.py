import os
import sys
import types
import logging
from types import SimpleNamespace

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.modules.setdefault("pyodbc", types.SimpleNamespace(Connection=object))

from enkibot.core.language_service import LanguageService


def test_missing_key_logs_event(caplog):
    ls = LanguageService(SimpleNamespace(), SimpleNamespace())
    ls.current_response_strings = {}
    with caplog.at_level(logging.ERROR):
        ls.get_response_string("nonexistent")
    assert any(getattr(r, "event_type", "") == "i18n.missing_key" for r in caplog.records)
