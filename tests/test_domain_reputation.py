import os
import sys
import types

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.modules.setdefault("pyodbc", types.SimpleNamespace(Connection=object))

from enkibot.modules.fact_check import get_domain_reputation


def test_domain_reputation_known_sources():
    assert "propaganda" in get_domain_reputation("tass.ru")
    assert "reputable" in get_domain_reputation("reuters.com")
    assert "unknown or neutral" in get_domain_reputation("example.com")
