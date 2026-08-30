"""core/experiment/runner.py — a rebound OpenSearch retriever keeps its analyser.

The rebind exists to point a registry-resolved retriever at a specific
corpus and Realm. It rebuilds the object, so every constructor argument it
does not name is silently reset to a default, and `language` decides which
stemmer BM25 applies for the life of the index.

The Configuration Report compares Arabic, Russian and English on the same
grid. A rebind that drops the language makes every non-Russian row either
wrong or refused, which is why this is held separately from the analyser
tests themselves.
"""
from __future__ import annotations

import pytest

from core.experiment.runner import _rebind_corpus_id


class _FakeOpenSearchRetriever:
    """Named to match, because the rebind dispatches on the class name."""

    def __init__(self, language: str) -> None:
        self._host = "os-host"
        self._port = 9200
        self._strategy_id = "fixed"
        self._language = language
        self._corpus_id = "default"
        self._realm_id = None


_FakeOpenSearchRetriever.__name__ = "OpenSearchRetriever"


@pytest.fixture
def rebuilt(monkeypatch):
    """Capture the constructor call instead of connecting to OpenSearch."""
    seen: dict = {}

    class _Spy:
        def __init__(self, **kw):
            seen.update(kw)

    import adapters.opensearch as os_mod
    monkeypatch.setattr(os_mod, "OpenSearchRetriever", _Spy)
    return seen


def test_language_survives_the_rebind(rebuilt):
    _rebind_corpus_id(_FakeOpenSearchRetriever("ar"), "miracl-ar", None, None, None)
    assert rebuilt["language"] == "ar"
    assert rebuilt["corpus_id"] == "miracl-ar"


def test_a_russian_retriever_stays_russian(rebuilt):
    _rebind_corpus_id(_FakeOpenSearchRetriever("ru_be"), "handbook", None, None, None)
    assert rebuilt["language"] == "ru_be"


def test_host_and_port_still_come_from_the_config_when_given(rebuilt):
    """The rebind's original job, held here so the added argument did not
    displace it."""
    _rebind_corpus_id(
        _FakeOpenSearchRetriever("en"), "c", "realm-1",
        None, {"host": "other", "port": 9201},
    )
    assert (rebuilt["host"], rebuilt["port"]) == ("other", 9201)
    assert rebuilt["realm_id"] == "realm-1"
    assert rebuilt["language"] == "en"
