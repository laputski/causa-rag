"""services/api_gateway/routers/experiments.py:_CompositeEvaluator — Eval
Measurement Trustworthiness, Phase 0. Replaces _BasicEvaluator's Jaccard
token-overlap (see core/eval/answerability.py for why that was unsound).

Uses a fake embedder (no model load) so these tests assert the GATING logic
(answerable vs uncovered/out_of_scope routes to different metric sets) —
not embedding quality, which is covered separately in test_semantic_metrics.py.
"""
from __future__ import annotations

from core.eval.ref_resolution import IndexRefResolver
from core.models import Answer, SourceRef
from services.api_gateway.routers.experiments import _CompositeEvaluator

# Coverage is resolved against an in-memory ref index instead of
# a real corpus directory on disk. The set below reproduces exactly what the
# old disk lookup found on disk for the refs these tests use, so
# every gating expectation is unchanged; the tests simply no longer need
# 42 MB of corpus present to run.
_REF_RESOLVER = IndexRefResolver(
    known_ref_ids=frozenset({"SRC001/1", "SRC001/44", "SRC001/47", "SRC001/210"})
)


class _FakeEmbedder:
    """Deterministic stand-in — only needs to support .embed() with a
    shape compatible with semantic_metrics' cosine math; exact values
    don't matter for these gating tests."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0, 0.0] for t in texts]


def _evaluator(top_k: int = 5) -> _CompositeEvaluator:
    return _CompositeEvaluator(embedder=_FakeEmbedder(), top_k=top_k, ref_resolver=_REF_RESOLVER)


def _answer(text: str, source_refs: list[SourceRef] | None = None) -> Answer:
    return Answer(text=text, source_refs=source_refs or [])


def test_answerable_question_gets_full_metric_set() -> None:
    question = {"question": "What may a team decide?", "article_refs": ["SRC001/44"], "ground_truth": "ref"}
    answer = _answer(
        "A team may decide within its own budget.",
        [SourceRef(doc_id="d1", chunk_id="c1", chunk_text="context", source_code="SRC001", article_no="44")],
    )
    metrics = _evaluator().evaluate(question, answer)
    # grounded_in_correct_source IS present here because the one source_ref
    # matches article_refs exactly — see the dedicated gating tests below
    # for the "no matching chunk" / "no reranker" cases where keys are absent.
    assert set(metrics) == {
        "correct_refusal", "retrieval_recall_at_k", "retrieval_precision_at_k",
        "retrieval_average_precision", "answer_similarity", "answer_relevance",
        "context_support", "grounded_in_correct_source",
    }


def test_uncovered_question_gets_only_correct_refusal() -> None:
    """A gap with no real ground truth to score retrieval
    or answer quality against, only whether the system correctly refused."""
    question = {"question": "What do the rules say about working hours?", "article_refs": ["SRC003/101"]}
    answer = _answer("The provided documents contain no information on this.")
    metrics = _evaluator().evaluate(question, answer)
    assert set(metrics) == {"correct_refusal"}


def test_out_of_scope_question_gets_only_correct_refusal() -> None:
    question = {"question": "What is today's exchange rate?", "article_refs": []}
    answer = _answer("Not found in the documents.")
    metrics = _evaluator().evaluate(question, answer)
    assert set(metrics) == {"correct_refusal"}


def test_answerable_question_with_real_answer_scores_correct_refusal_one() -> None:
    question = {"question": "Q", "article_refs": ["SRC001/44"], "ground_truth": "ref"}
    answer = _answer("A real, substantive answer.")
    metrics = _evaluator().evaluate(question, answer)
    assert metrics["correct_refusal"] == 1.0


def test_answerable_question_that_refuses_scores_correct_refusal_zero() -> None:
    """Refusing on an answerable question is a real retrieval/generation
    failure, not a correct decision — must score 0, not 1."""
    question = {"question": "Q", "article_refs": ["SRC001/44"], "ground_truth": "ref"}
    answer = _answer("No information in the documents.")
    metrics = _evaluator().evaluate(question, answer)
    assert metrics["correct_refusal"] == 0.0


def test_uncovered_question_that_refuses_scores_correct_refusal_one() -> None:
    question = {"question": "Q", "article_refs": ["SRC003/101"]}
    answer = _answer("Information not found.")
    metrics = _evaluator().evaluate(question, answer)
    assert metrics["correct_refusal"] == 1.0


def test_uncovered_question_that_hallucinates_scores_correct_refusal_zero() -> None:
    """Answering confidently on a source the corpus doesn't have is a
    hallucination, not a success — must score 0."""
    question = {"question": "Q", "article_refs": ["SRC003/101"]}
    answer = _answer("The rules guarantee a right to rest in article 101.")
    metrics = _evaluator().evaluate(question, answer)
    assert metrics["correct_refusal"] == 0.0


def test_empty_answer_text_counts_as_refusal() -> None:
    question = {"question": "Q", "article_refs": []}
    answer = _answer("")
    metrics = _evaluator().evaluate(question, answer)
    assert metrics["correct_refusal"] == 1.0


def test_retrieval_metrics_use_configured_top_k() -> None:
    question = {"question": "Q", "article_refs": ["SRC001/1"], "ground_truth": "ref"}
    refs = [
        SourceRef(doc_id="d1", chunk_id=f"c{i}", chunk_text="x", source_code="SRC001", article_no=f"9{i}")
        for i in range(10)
    ]
    refs[7] = SourceRef(doc_id="d1", chunk_id="c7", chunk_text="x", source_code="SRC001", article_no="1")
    answer = _answer("an answer", refs)
    assert _evaluator(top_k=5).evaluate(question, answer)["retrieval_recall_at_k"] == 0.0
    assert _evaluator(top_k=10).evaluate(question, answer)["retrieval_recall_at_k"] == 1.0


def test_missing_article_refs_key_is_treated_as_out_of_scope() -> None:
    question = {"question": "Q"}  # no article_refs key at all (legacy row)
    answer = _answer("Not found.")
    metrics = _evaluator().evaluate(question, answer)
    assert set(metrics) == {"correct_refusal"}
    assert metrics["correct_refusal"] == 1.0


# ── grounded_in_correct_source + pre_rerank_recall_at_k (Phase 1) ──────────────

def test_grounded_in_correct_source_absent_when_no_chunk_matches_refs() -> None:
    """recall_at_k=0 case — nothing to compute grounding against, key must
    be absent (not a misleading 0.0)."""
    question = {"question": "Q", "article_refs": ["SRC001/44"], "ground_truth": "ref"}
    answer = _answer(
        "An answer.",
        [SourceRef(doc_id="d1", chunk_id="c1", chunk_text="the wrong article", source_code="SRC001", article_no="626")],
    )
    metrics = _evaluator().evaluate(question, answer)
    assert "grounded_in_correct_source" not in metrics
    assert metrics["retrieval_recall_at_k"] == 0.0


def test_pre_rerank_recall_absent_when_no_reranker_ran() -> None:
    """answer.pre_rerank_source_refs is empty by construction when no
    reranker ran (core/pipeline.py) — the metric key must be absent, not 0.0."""
    question = {"question": "Q", "article_refs": ["SRC001/44"], "ground_truth": "ref"}
    answer = _answer(
        "An answer.",
        [SourceRef(doc_id="d1", chunk_id="c1", chunk_text="x", source_code="SRC001", article_no="44")],
    )
    metrics = _evaluator().evaluate(question, answer)
    assert "pre_rerank_recall_at_k" not in metrics


def test_pre_rerank_recall_present_and_differs_from_post_rerank_recall() -> None:
    """The exact scenario the funnel module needs: reranker demoted the
    correct chunk out of the final top-k. pre_rerank_recall_at_k must
    reflect the wider pre-rerank list, distinct from the final recall."""
    question = {"question": "Q", "article_refs": ["SRC001/44"], "ground_truth": "ref"}
    answer = _answer(
        "An answer.",
        source_refs=[SourceRef(doc_id="d1", chunk_id="c1", chunk_text="x", source_code="SRC001", article_no="99")],
    )
    answer.pre_rerank_source_refs = [
        SourceRef(doc_id="d1", chunk_id="c1", chunk_text="x", source_code="SRC001", article_no="44"),
        SourceRef(doc_id="d1", chunk_id="c2", chunk_text="y", source_code="SRC001", article_no="99"),
    ]
    metrics = _evaluator().evaluate(question, answer)
    assert metrics["retrieval_recall_at_k"] == 0.0
    assert metrics["pre_rerank_recall_at_k"] == 1.0


def test_citation_number_coverage_present_when_structural_path_has_a_label() -> None:
    """The actual production bug this metric exists to catch: model wrote
    the wrong article number despite retrieval finding the right chunk
    (retrieval_recall_at_k=1.0 below) — a metric that only looked at
    source_refs/retrieval would miss this entirely."""
    question = {"question": "Q", "article_refs": ["SRC001/210"], "ground_truth": "ref"}
    answer = _answer(
        "The escrow agent. Article 1.",
        [SourceRef(
            doc_id="d1", chunk_id="c1", chunk_text="x", source_code="SRC001", article_no="210",
            structural_path="document/article[Article 210. Closing an escrow account]",
        )],
    )
    metrics = _evaluator().evaluate(question, answer)
    assert metrics["retrieval_recall_at_k"] == 1.0
    assert metrics["citation_number_coverage"] == 0.0


def test_citation_number_coverage_absent_when_no_structural_label() -> None:
    question = {"question": "Q", "article_refs": ["SRC001/44"], "ground_truth": "ref"}
    answer = _answer(
        "An answer.",
        [SourceRef(doc_id="d1", chunk_id="c1", chunk_text="x", source_code="SRC001", article_no="44")],
    )
    metrics = _evaluator().evaluate(question, answer)
    assert "citation_number_coverage" not in metrics


# ── resolve_answerability (exposed for core/experiment/runner.py to persist
# onto QuestionResult.answerability — Found live: a question with NO
# article_refs at all (out_of_scope) displayed the same "reference source
# missing from corpus (uncovered)" funnel wording as a question whose refs
# genuinely don't resolve, because the display layer could only re-derive
# "answerable vs not" from metric-key presence, never the real sub-class.
# This method is what lets the real class get persisted instead of re-guessed.) ──

def test_resolve_answerability_matches_evaluate_for_answerable() -> None:
    question = {"question": "Q", "article_refs": ["SRC001/44"], "ground_truth": "ref"}
    assert _evaluator().resolve_answerability(question) == "answerable"


def test_resolve_answerability_matches_evaluate_for_uncovered() -> None:
    question = {"question": "Q", "article_refs": ["SRC003/101"]}
    assert _evaluator().resolve_answerability(question) == "uncovered"


def test_resolve_answerability_matches_evaluate_for_out_of_scope() -> None:
    """The exact real-world case this fix targets: a generated question with
    no article_refs at all — distinct from "uncovered" (refs exist but don't
    resolve), and the funnel wording must be able to tell them apart."""
    question = {"question": "Q", "article_refs": []}
    assert _evaluator().resolve_answerability(question) == "out_of_scope"


def test_resolve_answerability_respects_explicit_field_like_evaluate_does() -> None:
    question = {"question": "Q", "article_refs": [], "answerability": "answerable"}
    assert _evaluator().resolve_answerability(question) == "answerable"


# ── _display_answerability (GET /experiments/{run_id}'s funnel-diagnosis
# input selection) — same bug this whole section targets, at the display
# layer: it used to re-guess "answerable vs not" from metric-key presence
# alone, unable to recover "uncovered" vs "out_of_scope" for a stored run. ──

def test_display_answerability_prefers_the_persisted_field() -> None:
    from services.api_gateway.routers.experiments import _display_answerability
    qr = {"answerability": "out_of_scope", "metrics": {"correct_refusal": 0.0}}
    assert _display_answerability(qr, qr["metrics"]) == "out_of_scope"


def test_display_answerability_distinguishes_uncovered_from_out_of_scope() -> None:
    from services.api_gateway.routers.experiments import _display_answerability
    metrics = {"correct_refusal": 0.0}
    assert _display_answerability({"answerability": "uncovered"}, metrics) == "uncovered"
    assert _display_answerability({"answerability": "out_of_scope"}, metrics) == "out_of_scope"


def test_display_answerability_falls_back_to_metric_key_guess_for_historical_runs() -> None:
    """A run stored before QuestionResult.answerability existed has no such
    key at all — falls back to the old best-effort inference (which cannot
    distinguish uncovered from out_of_scope, same limitation as before)."""
    from services.api_gateway.routers.experiments import _display_answerability
    answerable_metrics = {"retrieval_recall_at_k": 1.0, "correct_refusal": 1.0}
    assert _display_answerability({}, answerable_metrics) == "answerable"
    non_answerable_metrics = {"correct_refusal": 0.0}
    assert _display_answerability({}, non_answerable_metrics) == "not_applicable"
