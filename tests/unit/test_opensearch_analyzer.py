"""The analyser follows the corpus's language, and a mismatch refuses to start.

Until now every index in every language was created with the Russian and
Belarusian analyser. Arabic text was stemmed by Russian rules and filtered
through Russian stop words; English was too. BM25 kept returning results, which
is what made it hard to see. The numbers were not low, they were meaningless,
and every hybrid merge inherited that.

The second half matters as much as the first. An index carries its analyser for
life. A query that arrives before ingestion creates the index with whatever the
caller asked for, ingestion finds it existing and leaves it alone, and the
corpus is then stemmed by the wrong language for as long as it lives. Nothing
would say so, which is why the guard raises instead of warning.
"""

from __future__ import annotations

import pytest

from adapters.opensearch import (
    DEFAULT_LANGUAGE,
    LANGUAGE_ANALYZERS,
    OpenSearchRetriever,
    index_settings,
)


class _Indices:
    """Just enough of the client to exercise index creation."""

    def __init__(self, existing: dict | None = None, readable: bool = True) -> None:
        self.existing = existing
        self.readable = readable
        self.created: dict | None = None

    def exists(self, index: str) -> bool:
        return self.existing is not None

    def create(self, index: str, body: dict) -> None:
        self.created = body

    def get_mapping(self, index: str) -> dict:
        if not self.readable:
            raise RuntimeError("mapping unavailable")
        return {index: {"mappings": {"properties": {"text": self.existing}}}}


def _retriever(monkeypatch, indices: _Indices, language: str) -> OpenSearchRetriever:
    class _Client:
        def __init__(self, *_args, **_kwargs) -> None:
            self.indices = indices

    monkeypatch.setattr("opensearchpy.OpenSearch", _Client)
    return OpenSearchRetriever(strategy_id="fixed", language=language)


class TestSettingsFollowTheLanguage:
    @pytest.mark.parametrize(
        ("language", "analyzer"),
        [("ar", "arabic"), ("en", "english"), ("ru", "russian"), ("fa", "persian")],
    )
    def test_a_language_gets_its_own_analyser(self, language: str, analyzer: str) -> None:
        settings = index_settings(language)
        assert settings["mappings"]["properties"]["text"]["analyzer"] == analyzer

    def test_the_default_is_unchanged(self) -> None:
        """An existing corpus keeps the behaviour it was indexed with."""
        text = index_settings()["mappings"]["properties"]["text"]
        assert DEFAULT_LANGUAGE == "ru_be"
        assert text["analyzer"] == "ru_be_analyzer"

    def test_the_custom_analyser_is_declared_and_a_built_in_is_not(self) -> None:
        """A built-in is referred to by name; only the custom one needs defining."""
        assert "analysis" in index_settings("ru_be")["settings"]
        assert "analysis" not in index_settings("ar")["settings"]

    def test_an_unknown_language_is_refused_rather_than_guessed(self) -> None:
        with pytest.raises(ValueError, match="unknown analyser language"):
            index_settings("klingon")

    def test_search_and_index_analysers_agree(self) -> None:
        """Different analysers on the two sides silently break every match."""
        for language in LANGUAGE_ANALYZERS:
            text = index_settings(language)["mappings"]["properties"]["text"]
            assert text["analyzer"] == text["search_analyzer"], language


class TestAMismatchRefusesToStart:
    def test_a_new_index_is_built_with_the_asked_for_analyser(self, monkeypatch) -> None:
        indices = _Indices(existing=None)
        _retriever(monkeypatch, indices, "ar")
        assert indices.created["mappings"]["properties"]["text"]["analyzer"] == "arabic"

    def test_an_index_built_for_another_language_raises(self, monkeypatch) -> None:
        indices = _Indices(existing={"analyzer": "ru_be_analyzer"})
        with pytest.raises(RuntimeError, match="keeps its analyser for life"):
            _retriever(monkeypatch, indices, "ar")

    def test_a_matching_index_is_left_alone(self, monkeypatch) -> None:
        indices = _Indices(existing={"analyzer": "arabic"})
        _retriever(monkeypatch, indices, "ar")
        assert indices.created is None

    def test_an_unreadable_mapping_does_not_refuse(self, monkeypatch) -> None:
        """Not knowing is not evidence of a mismatch.

        Refusing to start over a mapping that could not be read would be worse
        than the problem it guards against.
        """
        indices = _Indices(existing={"analyzer": "arabic"}, readable=False)
        _retriever(monkeypatch, indices, "ar")
        assert indices.created is None

    def test_an_index_with_no_analyser_named_counts_as_standard(self, monkeypatch) -> None:
        indices = _Indices(existing={"type": "text"})
        with pytest.raises(RuntimeError, match="'standard' analyser"):
            _retriever(monkeypatch, indices, "ar")


class TestLanguageCoverage:
    """The set was nine entries, and the README promises real analysers for
    documents that stop being English. Nine is not a choice anybody made."""

    def test_a_language_lucene_supports_is_not_refused(self) -> None:
        for language in ("italian", "portuguese", "hindi", "turkish", "thai", "cjk"):
            assert index_settings(language)["mappings"]["properties"]["text"]["analyzer"] == language

    def test_iso_codes_reach_the_same_analyser_as_the_lucene_name(self) -> None:
        """A corpus can be named the way its documents are tagged rather
        than the way Lucene spells the language."""
        for code, name in (("it", "italian"), ("pt", "portuguese"), ("hi", "hindi")):
            assert LANGUAGE_ANALYZERS[code] == LANGUAGE_ANALYZERS[name] == name

    def test_the_two_that_are_not_lucene_names_still_resolve(self) -> None:
        assert LANGUAGE_ANALYZERS["ru_be"] == "ru_be_analyzer"
        assert LANGUAGE_ANALYZERS["standard"] == "standard"

    def test_ru_stays_russian_rather_than_becoming_ru_be(self) -> None:
        """The bait. Folding the ISO code onto the project's own analyser
        would stem every Russian corpus with Belarusian stop words without
        anyone asking for it."""
        assert LANGUAGE_ANALYZERS["ru"] == "russian"

    def test_an_invented_language_is_still_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown analyser language"):
            index_settings("klingon")

    def test_only_ru_be_carries_a_custom_analysis_block(self) -> None:
        """A built-in is referenced by name; defining one under its own name
        would shadow Lucene's with an empty imitation."""
        assert index_settings("italian")["settings"].get("analysis") is None
        assert "ru_be_analyzer" in index_settings("ru_be")["settings"]["analysis"]["analyzer"]
