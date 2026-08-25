"""Generic answerability for external-RAG datasets.

resolve_answerability() must prefer an explicit per-row `answerability`
field over the platform's own coverage check, so a dataset belonging to a
RAG whose corpus the platform never sees is classified by its own author
rather than by an index it has no relationship to. Zero migration, a later change replaced the disk lookup these tests originally exercised with a
core/eval/ref_resolution.py resolver over the index. The guarantee is
unchanged and still pinned here; only what "the platform's own check"
means underneath has changed.
"""
from __future__ import annotations

from core.eval.answerability import resolve_answerability
from core.eval.ref_resolution import IndexRefResolver, UnknownRefResolver

_INDEXED = IndexRefResolver(known_ref_ids=frozenset({"doc1/5"}))


def test_explicit_field_wins_over_the_platform_check() -> None:
    question = {"article_refs": ["nonexistent/1"], "answerability": "answerable"}
    assert resolve_answerability(question, _INDEXED) == "answerable"


def test_invalid_explicit_field_falls_back_to_the_platform_check() -> None:
    question = {"article_refs": [], "answerability": "not-a-real-class"}
    assert resolve_answerability(question, _INDEXED) == "out_of_scope"


def test_falls_back_to_the_index_when_field_absent() -> None:
    assert resolve_answerability({"article_refs": ["doc1/5"]}, _INDEXED) == "answerable"


def test_uncovered_via_the_index_when_field_absent() -> None:
    assert resolve_answerability({"article_refs": ["missing-doc/1"]}, _INDEXED) == "uncovered"


def test_no_resolver_and_no_field_defaults_by_presence_of_refs() -> None:
    # An external RAG whose dataset carries neither: trust the refs rather
    # than inventing a coverage verdict the platform cannot support.
    assert resolve_answerability({"article_refs": ["x/1"]}, None) == "answerable"
    assert resolve_answerability({"article_refs": []}, None) == "out_of_scope"
    assert resolve_answerability({}, None) == "out_of_scope"


def test_unavailable_resolver_never_reports_uncovered() -> None:
    question = {"article_refs": ["doc1/5"]}
    assert resolve_answerability(question, UnknownRefResolver()) == "answerable"
