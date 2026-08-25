"""core/eval/semantic_metrics.py — Eval Measurement Trustworthiness, Phase 0.

Uses BgeM3Embedder in stub mode (deterministic hash-based vectors, no GPU)
— sufficient to test the metric math (cosine, max-vs-mean) without pulling
in a real model. Stub vectors are NOT semantically meaningful (random per
distinct string), so these tests only assert properties that hold
regardless of embedding quality: identity, symmetry, range, and that
context_support actually takes the max rather than averaging.
"""
from __future__ import annotations

from adapters.bge_m3 import BgeM3Embedder
from core.eval.semantic_metrics import (
    _cosine,
    answer_relevance,
    answer_similarity,
    context_support,
    grounded_in_correct_source,
)


def test_cosine_identical_vectors_is_one() -> None:
    v = [0.5, 0.5, 0.0]
    assert abs(_cosine(v, v) - 1.0) < 1e-9


def test_cosine_orthogonal_vectors_is_zero() -> None:
    assert abs(_cosine([1.0, 0.0], [0.0, 1.0])) < 1e-9


def test_cosine_zero_vector_returns_zero_not_nan() -> None:
    assert _cosine([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_answer_similarity_identical_text_is_near_one() -> None:
    embedder = BgeM3Embedder()
    text = "A legal entity is an organisation that owns property in its own right."
    assert answer_similarity(text, text, embedder) > 0.999


def test_answer_similarity_is_symmetric() -> None:
    embedder = BgeM3Embedder()
    a = "One answer."
    b = "A completely different answer about something else."
    assert abs(answer_similarity(a, b, embedder) - answer_similarity(b, a, embedder)) < 1e-9


def test_answer_similarity_empty_strings_return_zero() -> None:
    embedder = BgeM3Embedder()
    assert answer_similarity("", "something", embedder) == 0.0
    assert answer_similarity("something", "", embedder) == 0.0
    assert answer_similarity("", "", embedder) == 0.0


def test_context_support_picks_max_not_mean() -> None:
    """If one chunk is identical to the answer and the rest are irrelevant,
    context_support must reflect the identical chunk (~1.0), not be diluted
    by averaging across irrelevant chunks — the dilution bug in the old
    Jaccard-over-the-whole-context metric this replaces."""
    embedder = BgeM3Embedder()
    answer = "The limitation period is three years."
    chunks = [
        "Entirely unrelated text about vehicle leasing.",
        "Another irrelevant fragment about company reorganisation.",
        answer,  # identical chunk
        "And one more irrelevant fragment.",
    ]
    score = context_support(answer, chunks, embedder)
    assert score > 0.999


def test_context_support_empty_context_returns_zero() -> None:
    embedder = BgeM3Embedder()
    assert context_support("some answer", [], embedder) == 0.0


def test_context_support_empty_answer_returns_zero() -> None:
    embedder = BgeM3Embedder()
    assert context_support("", ["some context"], embedder) == 0.0


# ── answer_relevance (deterministic counterpart to deepeval AnswerRelevancy) ──

def test_answer_relevance_identical_to_question_is_near_one() -> None:
    embedder = BgeM3Embedder()
    text = "Does article 5 apply by analogy?"
    assert answer_relevance(text, text, embedder) > 0.999


def test_answer_relevance_is_symmetric() -> None:
    embedder = BgeM3Embedder()
    a = "What is the limitation period?"
    b = "A wholly unrelated answer about leasing."
    assert abs(answer_relevance(a, b, embedder) - answer_relevance(b, a, embedder)) < 1e-9


def test_answer_relevance_empty_strings_return_zero() -> None:
    embedder = BgeM3Embedder()
    assert answer_relevance("", "a question?", embedder) == 0.0
    assert answer_relevance("an answer", "", embedder) == 0.0
    assert answer_relevance("", "", embedder) == 0.0


# ── grounded_in_correct_source (Eval Measurement Trustworthiness, Phase 1) ─────

def test_grounded_in_correct_source_uses_only_matching_chunk() -> None:
    """The exact scenario this metric exists for (run 44329138): a chunk
    that LOOKS relevant (high context_support) but isn't the ground-truth
    article must not inflate this metric — only the chunk matching
    article_refs counts."""
    embedder = BgeM3Embedder()
    answer = "A legal entity owns property in its own right."
    source_refs = [
        {"chunk_text": "A completely different article about leasing.", "source_code": "SRC001", "article_no": "626"},
        {"chunk_text": answer, "source_code": "SRC001", "article_no": "44"},  # the correct one, identical text
    ]
    score = grounded_in_correct_source(answer, source_refs, ["SRC001/44"], embedder)
    assert score is not None
    assert score > 0.999


def test_grounded_in_correct_source_none_when_no_chunk_matches_refs() -> None:
    """recall_at_k=0 case — the correct article wasn't retrieved at all.
    Must return None (undefined), not a misleading 0.0 (which would read as
    "confidently wrong" rather than "nothing to measure against")."""
    embedder = BgeM3Embedder()
    source_refs = [
        {"chunk_text": "Something wrong.", "source_code": "SRC001", "article_no": "626"},
    ]
    assert grounded_in_correct_source("an answer", source_refs, ["SRC001/44"], embedder) is None


def test_grounded_in_correct_source_none_for_empty_answer() -> None:
    embedder = BgeM3Embedder()
    source_refs = [{"chunk_text": "some text", "source_code": "SRC001", "article_no": "44"}]
    assert grounded_in_correct_source("", source_refs, ["SRC001/44"], embedder) is None


def test_grounded_in_correct_source_none_for_empty_article_refs() -> None:
    embedder = BgeM3Embedder()
    source_refs = [{"chunk_text": "some text", "source_code": "SRC001", "article_no": "44"}]
    assert grounded_in_correct_source("an answer", source_refs, [], embedder) is None


# ── Corpus with no external document-code numbering scheme (found live: this
# metric silently vanished — None — for EVERY question of one Realm's run, since
# matching used to hardcode "{source_code}/{article_no}" and source_code is
# never present on this kind of corpus's chunks at all) ──────────────────────

def test_grounded_in_correct_source_matches_doc_id_structural_path_refs() -> None:
    embedder = BgeM3Embedder()
    answer = "The unit must be switched off before servicing."
    source_refs = [
        {"chunk_text": "Unrelated text.", "doc_id": "d1", "structural_path": "Another section"},
        {"chunk_text": answer, "doc_id": "d2", "structural_path": "Servicing"},
    ]
    score = grounded_in_correct_source(answer, source_refs, ["d2#Servicing"], embedder)
    assert score is not None
    assert score > 0.999


def test_grounded_in_correct_source_matches_bare_doc_id_refs() -> None:
    embedder = BgeM3Embedder()
    answer = "An answer from the document."
    source_refs = [{"chunk_text": answer, "doc_id": "d2"}]
    score = grounded_in_correct_source(answer, source_refs, ["d2"], embedder)
    assert score is not None
    assert score > 0.999
