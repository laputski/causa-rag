"""Regression guard + silent-degradation detectors."""
from __future__ import annotations

from core.eval import detectors as det
from core.eval.regression import compare

# ── regression guard ───────────────────────────────────────────────────────────

def test_no_regression_when_metrics_hold():
    rep = compare({"faithfulness": 0.80}, {"faithfulness": 0.80})
    assert rep.passed
    assert not rep.violations


def test_regression_flagged_on_drop():
    rep = compare({"faithfulness": 0.50}, {"faithfulness": 0.80})
    assert not rep.passed
    assert any(d.metric == "faithfulness" and d.regressed for d in rep.deltas)


def test_sla_absolute_floor():
    rep = compare({"accuracy_semantic": 0.70}, {"accuracy_semantic": 0.71})
    assert not rep.passed  # below 0.85 floor even though only a small drop vs baseline


def test_small_drop_within_threshold_passes():
    rep = compare({"faithfulness": 0.78}, {"faithfulness": 0.80})  # 2.5% < 5%
    assert rep.passed


# ── detectors ──────────────────────────────────────────────────────────────────

def _run(refs=None, answers=None):
    qrs = []
    if refs is not None:
        qrs.append({"generated_answer": "a", "source_refs": refs})
    for a in answers or []:
        qrs.append({"generated_answer": a, "source_refs": []})
    return {"question_results": qrs}


def test_detect_stub_embedder():
    refs = [{"dense_score": 0.0, "chunk_text": f"t{i}"} for i in range(10)]
    assert det.detect_stub_embedder(refs) is not None
    good = [{"dense_score": 0.6, "chunk_text": f"t{i}"} for i in range(10)]
    assert det.detect_stub_embedder(good) is None


def test_detect_duplicates():
    """The same text twice in one question's context, and never across a run.

    The old assertion handed it a flat list of refs, which is how the detector
    read a whole run: every question's sources pooled together. There a chunk
    that answers two questions is indistinguishable from a chunk stored twice,
    so the rule measured how useful a chunk is and called that a defect.

    Measured on the proving ground's healthy corpus: fifteen questions, seventy
    five sources, twenty-eight distinct texts, the most useful chunk answering
    six questions, and not one repeat inside any single question. The pooled
    rule reported forty-seven duplicates. Over the forty-one stored runs it
    fired on thirty-two and was wrong on all thirty-two; it now fires on none.
    """
    def run(*contexts):
        return {"question_results": [
            {"source_refs": [{"chunk_text": t} for t in texts]} for texts in contexts
        ]}

    assert det.detect_duplicates(run(["same", "same", "other"])) is not None
    assert det.detect_duplicates(run(["a", "b"])) is None


def test_a_chunk_answering_two_questions_is_not_a_duplicate():
    """The bait for the rule above, and the false positive it was written for.

    A small corpus with a well-chosen question set produces this on purpose:
    the passage that answers "how often" also answers "who may". Reporting it
    as a re-ingest artefact told the reader to drop the collection and reindex
    a corpus that was correct."""
    shared = {"question_results": [
        {"source_refs": [{"chunk_text": "the calibration interval"}, {"chunk_text": "a"}]},
        {"source_refs": [{"chunk_text": "the calibration interval"}, {"chunk_text": "b"}]},
    ]}
    assert det.detect_duplicates(shared) is None


def test_the_count_reported_is_of_repeats_inside_contexts() -> None:
    """A count pooled across questions would be larger and would mean nothing
    a reader could act on."""
    item = det.detect_duplicates({"question_results": [
        {"source_refs": [{"chunk_text": "x"}, {"chunk_text": "x"}, {"chunk_text": "x"}]},
        {"source_refs": [{"chunk_text": "y"}, {"chunk_text": "y"}]},
        {"source_refs": [{"chunk_text": "z"}]},
    ]})
    assert item is not None
    assert "3 repeated chunks" in item.detail
    assert "2 question(s)" in item.detail


def test_detect_header_only():
    refs = [{"structural_path": "section[X]", "chunk_text": "X"} for _ in range(4)]
    assert det.detect_header_only(refs) is not None
    full = [{"structural_path": "section[X]", "chunk_text": "a long, substantive stretch of article text " * 3}]
    assert det.detect_header_only(full) is None


def test_detect_bm25_dominance():
    """Where the chunks came from, not how large their scores are.

    The old assertion was `dense_score=0.1, sparse_score=0.9` fires. It cannot
    any more, and that is the fix: those two numbers are on different scales,
    and comparing their magnitudes reported dominance on 100% of the 4664
    scored pairs stored on this machine, on every hybrid run. Contribution to
    the final context is scale-free and is what the name claims to measure.
    """
    keyword_only = [{"dense_score": 0.0, "sparse_score": 0.9} for _ in range(25)]
    assert det.detect_bm25_dominance(keyword_only) is not None

    # Symmetric: a semantic half supplying everything is the same failure. The
    # old rule was structurally unable to see this direction, and one stored
    # run has it — 89% of its context from the semantic half alone.
    semantic_only = [{"dense_score": 0.7, "sparse_score": 0.0} for _ in range(25)]
    assert det.detect_bm25_dominance(semantic_only) is not None

    balanced = [
        {"dense_score": 0.5, "sparse_score": 0.0 if i % 3 else 4.0} for i in range(25)
    ]
    assert det.detect_bm25_dominance(balanced) is None

    # Below the minimum number of observations the balance of a merge means
    # nothing: one question's top-k would decide it.
    assert det.detect_bm25_dominance(keyword_only[:5]) is None


def test_detect_empty_answers():
    assert det.detect_empty_answers(["", "The documents contain no information", "a fine answer"]) is not None
    assert det.detect_empty_answers(["answer one", "answer two", "answer three"]) is None


def test_run_detectors_aggregates():
    """The refs carry a recorded score split, which is what makes the embedder
    check answerable at all.

    They used to carry `dense_score: 0.0` and nothing else, and the assertion
    was that this fires `stub_embedder`. It no longer does, and the change is a
    fix and not a regression: a payload where neither signal is reported is
    what a healthy dense-only run looks like, not what a stubbed corpus looks
    like. See the test below for what the two are told apart by now.
    """
    run = _run(refs=[{"dense_score": 0.0, "sparse_score": 3.0, "chunk_text": "x"} for _ in range(10)])
    items = det.run_detectors(run)
    assert any(i.id == "stub_embedder" for i in items)
    assert all(hasattr(i, "severity") for i in items)


def test_run_detectors_skips_stub_embedder_and_bm25_dominance_for_external_rag_runs():
    """Found live: an external RAG never reports a per-signal
    dense/sparse split at all — those fields land on SourceRef's plain-float
    defaults (0.0), which detect_stub_embedder/detect_bm25_dominance can't
    tell apart from a genuinely-collapsed dense score. Fired "5/5 chunks
    dense_score≈0" against a real, working answer whose actual (fused)
    score was a healthy 6.29.

    The behaviour still holds and the mechanism behind it changed. The embedder
    check no longer asks what produced the run; it asks whether the run
    recorded a split, which is the same question this special case stood in
    for, put to the data instead of to the configuration. It therefore also
    covers the dense-only in-process case, which the special case never did."""
    refs = [{"dense_score": 0.0, "sparse_score": 0.0, "chunk_text": f"t{i}"} for i in range(5)]
    run = {"config": {"pipeline_source": "http"}, "question_results": [
        {"generated_answer": "a", "source_refs": refs},
    ]}
    items = det.run_detectors(run)
    assert not any(i.id in ("stub_embedder", "bm25_dominance") for i in items)


def test_a_dense_only_run_is_not_accused_of_a_stub_embedder():
    """The premise this test used to assert was false, and eighteen stored runs
    say so.

    It read: "the platform's own retrieval DOES populate dense_score
    meaningfully" for an in-process run, so the detector must fire on a payload
    where every dense score is zero. Measured against the runs on disk: only
    the hybrid retriever ever writes that field. Every one of the eighteen
    stored dense-only runs carries zero in it for every chunk, and every one of
    them has a recall against its reference sources between 0.81 and 1.00. The
    detector reported "indexed with random vectors" at severity error on all
    eighteen.

    What separates the two cases is whether the run recorded a per-signal split
    at all. The pipeline's name says nothing about it.
    """
    refs = [{"dense_score": 0.0, "sparse_score": 0.0, "chunk_text": f"t{i}"} for i in range(5)]
    run = {"config": {"pipeline_source": "in_process"}, "question_results": [
        {"generated_answer": "a", "source_refs": refs},
    ]}
    ids = {i.id for i in det.run_detectors(run)}
    assert "stub_embedder" not in ids
    assert "embedder_unverified" in ids, (
        "silence would let a reader take the absence of a warning for a clean "
        "bill of health; the platform has to say it could not check"
    )


def test_a_collapsed_dense_side_beside_a_live_sparse_one_is_still_caught():
    """The fix must not turn a false alarm into a permanent silence."""
    refs = [{"dense_score": 0.0, "sparse_score": 2.5, "chunk_text": f"t{i}"} for i in range(5)]
    run = {"config": {"pipeline_source": "in_process"}, "question_results": [
        {"generated_answer": "a", "source_refs": refs},
    ]}
    assert any(i.id == "stub_embedder" for i in det.run_detectors(run))


def test_empty_answers_is_not_reported_for_a_run_that_skipped_generation():
    """A retrieval-only run answers nothing on purpose.

    Found the same way: the detector reported "100% of answers are empty" at
    severity error on every one of the thirty-six stored retrieval-only runs.
    The evaluator beside it already refuses to score an empty answer for
    exactly this reason.
    """
    run = {
        "config": {"retrieval_only": True},
        "question_results": [{"generated_answer": "", "source_refs": []} for _ in range(10)],
    }
    assert not any(i.id == "empty_answers" for i in det.run_detectors(run))

    generating = {
        "config": {"retrieval_only": False},
        "question_results": [{"generated_answer": "", "source_refs": []} for _ in range(10)],
    }
    assert any(i.id == "empty_answers" for i in det.run_detectors(generating))


# ── detect_incorrect_refusals (Eval Measurement Trustworthiness, Phase 0) ──────

def _run_with_correct_refusal(scores: list[float]) -> dict:
    return {"question_results": [{"metrics": {"correct_refusal": s}} for s in scores]}


def test_detect_incorrect_refusals_flags_high_wrong_rate():
    run = _run_with_correct_refusal([0.0, 0.0, 0.0, 1.0, 1.0])
    item = det.detect_incorrect_refusals(run)
    assert item is not None
    assert item.id == "incorrect_refusals"
    assert item.severity == "error"  # 60% wrong >= 40% error threshold


def test_detect_incorrect_refusals_passes_when_mostly_correct():
    run = _run_with_correct_refusal([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])  # 1/6 ≈ 17% < 20%
    assert det.detect_incorrect_refusals(run) is None


def test_detect_incorrect_refusals_none_when_metric_absent():
    """Runs evaluated before Phase 0 (or non-control-question runs) have no
    correct_refusal metric at all — must not false-alarm on missing data."""
    run = {"question_results": [{"metrics": {"faithfulness": 0.5}}]}
    assert det.detect_incorrect_refusals(run) is None


def test_run_detectors_includes_incorrect_refusals_when_present():
    run = _run_with_correct_refusal([0.0, 0.0, 0.0, 1.0])
    items = det.run_detectors(run)
    assert any(i.id == "incorrect_refusals" for i in items)


# ── detect_layer_bottleneck (Eval Measurement Trustworthiness, Phase 1) ────────

def _run_with_question_metrics(metrics_list: list[dict]) -> dict:
    return {"question_results": [{"metrics": m} for m in metrics_list]}


def test_detect_layer_bottleneck_flags_dominant_retrieval_failures():
    run = _run_with_question_metrics([
        {"retrieval_recall_at_k": 0.0, "answer_similarity": 0.1, "correct_refusal": 1.0}
        for _ in range(4)
    ] + [
        {"retrieval_recall_at_k": 1.0, "answer_similarity": 0.9, "correct_refusal": 1.0}
        for _ in range(1)
    ])
    item = det.detect_layer_bottleneck(run)
    assert item is not None
    assert item.id == "layer_bottleneck"
    assert "retrieval" in item.title


def test_detect_layer_bottleneck_flags_suspected_ungrounded_answers():
    """Mirrors the actual pattern found in run 44329138."""
    run = _run_with_question_metrics([
        {"retrieval_recall_at_k": 0.0, "answer_similarity": 0.85, "correct_refusal": 1.0}
        for _ in range(3)
    ] + [
        {"retrieval_recall_at_k": 1.0, "answer_similarity": 0.9, "correct_refusal": 1.0}
        for _ in range(2)
    ])
    item = det.detect_layer_bottleneck(run)
    assert item is not None
    assert "ungrounded" in item.title


def test_detect_layer_bottleneck_none_when_mostly_ok():
    run = _run_with_question_metrics([
        {"retrieval_recall_at_k": 1.0, "answer_similarity": 0.9, "correct_refusal": 1.0}
        for _ in range(10)
    ])
    assert det.detect_layer_bottleneck(run) is None


def test_detect_layer_bottleneck_none_when_no_answerable_questions():
    """All questions are uncovered/out_of_scope (no retrieval_recall_at_k
    key at all) — nothing to attribute, must not false-alarm."""
    run = _run_with_question_metrics([{"correct_refusal": 1.0} for _ in range(5)])
    assert det.detect_layer_bottleneck(run) is None


def test_detect_layer_bottleneck_distinguishes_rerank_from_retrieval():
    run = _run_with_question_metrics([
        {
            "retrieval_recall_at_k": 0.0, "pre_rerank_recall_at_k": 1.0,
            "answer_similarity": 0.2, "correct_refusal": 1.0,
        }
        for _ in range(4)
    ] + [{"retrieval_recall_at_k": 1.0, "answer_similarity": 0.9, "correct_refusal": 1.0}])
    item = det.detect_layer_bottleneck(run)
    assert item is not None
    assert "rerank" in item.title


def test_run_detectors_includes_layer_bottleneck_when_present():
    run = _run_with_question_metrics([
        {"retrieval_recall_at_k": 0.0, "answer_similarity": 0.1, "correct_refusal": 1.0}
        for _ in range(5)
    ])
    items = det.run_detectors(run)
    assert any(i.id == "layer_bottleneck" for i in items)


# ── Unverified coverage must be visible, not silent ───────────
# the design notes, task 4. Before this, a run whose
# answerability was never verified against the index looked identical to one
# that was — and an unverifiable corpus silently classified every question
# "uncovered", dropping it out of retrieval metrics with nothing to show for
# it. The whole point of the flag is that a reader can tell the two apart.

def test_unverified_coverage_is_reported() -> None:
    from core.eval.detectors import detect_unverified_coverage

    item = detect_unverified_coverage(
        {"coverage_check": {"checked": False, "reason": "index unreachable: boom"}}
    )
    assert item is not None
    assert item.severity == "warn"
    assert "index unreachable: boom" in item.detail


def test_verified_coverage_reports_nothing() -> None:
    from core.eval.detectors import detect_unverified_coverage

    assert detect_unverified_coverage({"coverage_check": {"checked": True, "reason": ""}}) is None


def test_run_stored_before_the_field_existed_reports_nothing() -> None:
    """Absence of the field is not evidence of a problem — old runs must not
    all start showing a warning they have no data to justify."""
    from core.eval.detectors import detect_unverified_coverage

    assert detect_unverified_coverage({}) is None
    assert detect_unverified_coverage({"coverage_check": {}}) is None


def test_unverified_coverage_reaches_run_detectors_output() -> None:
    from core.eval.detectors import run_detectors

    ids = [d.id for d in run_detectors({
        "question_results": [],
        "coverage_check": {"checked": False, "reason": "no resolver configured"},
    })]
    assert "unverified_coverage" in ids
