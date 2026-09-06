"""A guard nobody tried to fool is worth nothing.

Every entry in `core/eval/atlas.py` that claims a failure is detected is a
claim about a signal, and this suite is what makes the claim cost something.
Each bait is a pair: the signal must fire on a payload carrying the defect, and
must stay silent on one without it. Half a bait proves half of nothing: a
detector that fires on everything passes the first half.

Two decisions shape the file.

**One clean baseline per shape of data, not one per failure.** Every infected
payload is that baseline with exactly one change. Per-failure baselines drift
into tripping each other's detectors, and then "silent on the healthy one"
fails for reasons unrelated to the failure under test. The baseline itself is
asserted to produce no finding at all, so any firing below is attributable.

**Sizes come from the detectors' own thresholds.** Several signals carry a
minimum before they will speak: three score pairs, two distinct numbers, a
fifth of the applicable questions. A payload under the threshold is silent, and
that silence is indistinguishable from health, so it would make a bait pass
while proving nothing.

The suite runs in CI with no infrastructure. Signals that cannot be reached
without a live index or without the interface are not faked here; they are
declared as belonging to the proving-ground level, and a separate test makes
sure such a declaration is present and not assumed.
"""
from __future__ import annotations

from typing import Any

import pytest

from core.citation import citation_number_coverage
from core.eval.atlas import FAILURES
from core.eval.corpus_health import analyze
from core.eval.detectors import run_detectors
from core.eval.funnel import diagnose_question
from core.eval.root_cause import classify_retrieval_cause
from core.models import SourceRef

# Signals reachable only with a live index, or computed in the interface. Named
# here and not skipped silently: an unreachable bait is a fact about the
# platform, not a gap in this file.
# Their baits live in `tests/proving_ground/`, not here. Implementations for
# them stood in this file for a while and never ran: every signal they name is
# unreachable at this level, so the parametrised test skipped and the code
# below it was dead. Dead bait code is worse than none, because the file's
# length suggests a coverage it does not have.
STAND_ONLY = {
    "health:near_duplicates": "needs a live vector store to search neighbours",
    "ui:dense_score_low": "computed in the interface, baited by ui/src/test",
    "metric:correct_refusal": "computed by the evaluator, which needs an embedder",
    "funnel:rerank": "needs a run where a reranker actually ran",
}


# ── the clean baselines ───────────────────────────────────────────────────────

def _healthy_ref(i: int) -> dict[str, Any]:
    return {
        "doc_id": f"handbook/{i}",
        "chunk_id": f"chunk-{i}",
        "structural_path": f"document/article[{i}. Section {i}]",
        "chunk_text": f"Section {i} sets out the procedure that applies in this case, "
                      f"with the thresholds and the approver named in full.",
        "score": 0.7,
        "dense_score": 0.6,
        "sparse_score": 0.4,
        "source_code": "handbook",
        "article_no": str(i),
    }


def clean_run(n_questions: int = 20) -> dict[str, Any]:
    """A run on which no detector has anything to say."""
    return {
        "config": {"pipeline_id": "hybrid_rrf", "retrieval_only": False, "top_k": 5},
        "coverage_check": {"checked": True, "reason": ""},
        "question_results": [
            {
                "question_id": f"q{i}",
                "question": f"What does section {i} require?",
                "generated_answer": f"Section {i} requires the approver named there to sign off.",
                "answerability": "answerable",
                "metrics": {
                    "retrieval_recall_at_k": 1.0,
                    "answer_similarity": 0.8,
                    "grounded_in_correct_source": 0.8,
                    "correct_refusal": 1.0,
                },
                "source_refs": [_healthy_ref(i), _healthy_ref(i + 100)],
            }
            for i in range(1, n_questions + 1)
        ],
    }


def clean_chunks(n: int = 20) -> list[dict[str, Any]]:
    """A corpus on which corpus health has nothing to say."""
    return [
        {
            "text": f"Section {i} sets out the procedure that applies in this case, "
                    f"naming the approver and the threshold in full so the reader "
                    f"need not look elsewhere. Case number {i} is the worked example.",
            "structural_path": f"document/article[{i}. Section {i}]",
            "chunk_id": f"chunk-{i}",
        }
        for i in range(1, n + 1)
    ]


def clean_dense_only_run(n_questions: int = 20) -> dict[str, Any]:
    """A healthy run of a dense-only pipeline.

    A second baseline is needed because `dense_score` is written only by the
    hybrid retriever: a dense-only pipeline leaves it at the field's default,
    and the *retrieval score* lives in `score` instead. A payload of this shape
    is not a defect, it is what half the stored runs on this machine look like,
    and a detector reading the absent field as evidence has no way to tell them
    from a corpus indexed with random vectors.
    """
    run = clean_run(n_questions)
    run["config"] = {"pipeline_id": "naive", "retrieval_only": False, "top_k": 5}
    for qr in run["question_results"]:
        for ref in qr["source_refs"]:
            ref["dense_score"] = 0.0
            ref["sparse_score"] = 0.0
    return run


def clean_retrieval_only_run(n_questions: int = 20) -> dict[str, Any]:
    """A healthy run that stopped before the generator on purpose.

    The empty answers are the documented intent of the mode, not a failure of
    it. The evaluator already guards its own metrics against exactly this
    shape; a detector that does not is reading a deliberate silence as a fault.
    """
    run = clean_run(n_questions)
    run["config"] = {"pipeline_id": "hybrid_rrf", "retrieval_only": True, "top_k": 5}
    for qr in run["question_results"]:
        qr["generated_answer"] = ""
        qr["metrics"] = {"retrieval_recall_at_k": 1.0}
    return run


def test_the_clean_run_baseline_is_actually_clean() -> None:
    """The baseline every infected payload is derived from. If this fires, every
    bait below rests on a payload that was never healthy."""
    fired = [d.id for d in run_detectors(clean_run())]
    assert fired == [], f"the clean baseline is not clean: {fired}"


def test_the_clean_corpus_baseline_is_actually_clean() -> None:
    fired = [i.id for i in analyze(clean_chunks()).items if i.id != "ok"]
    assert fired == [], f"the clean corpus baseline is not clean: {fired}"


def test_two_runs_differing_in_nothing_are_called_comparable() -> None:
    """The third baseline, and it was missing until a comparison signal
    needed one. A warning fired on two identical runs would have made every
    bait in this file pass while proving nothing about it."""
    fired = sorted(_compare_ids())
    assert fired == [], f"two identical runs are reported as incomparable: {fired}"


# ── the baits ─────────────────────────────────────────────────────────────────
#
# Each returns the set of signal ids that fired. The harness compares it
# against the clean baseline's own set, so a bait asserts a difference rather
# than an absolute, and a detector that fires on everything cannot pass.

def _detector_ids(run: dict[str, Any]) -> set[str]:
    return {f"detector:{d.id}" for d in run_detectors(run)}


def _health_ids(chunks: list[dict[str, Any]]) -> set[str]:
    return {f"health:{i.id}" for i in analyze(chunks).items if i.id != "ok"}


def _compare_ids() -> set[str]:
    """What comparability says about two runs that differ in nothing.

    The silent half for a signal of this kind, and it was missing: the quiet
    baseline was built out of detectors and corpus health alone, so a
    comparison signal firing on two identical runs would have passed every
    bait in this file.
    """
    from core.experiment.compare import check_comparability
    from core.experiment.config import ComponentRef, ExperimentConfig
    from core.experiment.runner import ExperimentResult, QuestionResult

    def _result() -> ExperimentResult:
        result = ExperimentResult(
            config=ExperimentConfig(
                name="r", corpus_id="handbook", dataset_name="handbook.v1.jsonl",
                chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
                embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
                generator=ComponentRef(kind="generator", component_id="ollama"),
            ),
            dataset_name="handbook.v1.jsonl",
        )
        # Scattered on purpose. Identical values per question give a
        # resolution of zero, and a check about what a set can resolve can
        # then never speak on this baseline, which would make its silence
        # here prove nothing at all.
        scores = [0.2, 0.9, 0.5, 0.8, 0.3, 0.95, 0.4, 0.7, 0.6, 0.85] * 2
        result.question_results = [
            QuestionResult(question_id=f"q{i}", question="?", reference_answer="",
                           generated_answer="an answer",
                           metrics={"retrieval_recall_at_k": 1.0, "answer_similarity": score})
            for i, score in enumerate(scores, start=1)
        ]
        result.aggregate_metrics = {
            "retrieval_recall_at_k": 1.0,
            "answer_similarity": sum(scores) / len(scores),
        }
        result.corpus_manifest = {
            "embedder_id": "bge_m3", "embedder_version": "1.0.0",
            "document_count": 20, "documents_digest": "a" * 64,
            "loaded_at": "2026-09-01T10:00:00Z",
        }
        return result

    return {f"compare:{w.id}" for w in check_comparability(_result(), _result())}


def bait_F01_reingest_duplicates() -> set[str]:
    """Re-ingestion added instead of updating: the same chunk twice."""
    run = clean_run()
    run["question_results"][0]["source_refs"].append(_healthy_ref(1))
    chunks = clean_chunks()
    chunks.append(dict(chunks[0]))
    return _detector_ids(run) | _health_ids(chunks)


def bait_F04_export_lost_documents() -> set[str]:
    """A gap in an otherwise sequential numbering: the unit exists in the
    source and never reached the index."""
    chunks = [c for c in clean_chunks() if "[7." not in c["structural_path"]]
    return _health_ids(chunks)


def bait_F06_chunks_too_small() -> set[str]:
    chunks = [{**c, "text": "See above."} for c in clean_chunks()]
    return _health_ids(chunks)


def bait_F07_grounds_spread_across_chunks() -> set[str]:
    """The unit is indexed, is not found even at a wide k, and occupies several
    chunks: no single piece carries enough of it."""
    cause = classify_retrieval_cause(
        presences={"handbook/7": "present"},
        miss={"found": False, "widened_k": 200},
        top_k=5,
        chunk_counts={"handbook/7": 6},
    )
    return {f"cause:{cause.cause}"}


def bait_F10_stub_embedder() -> set[str]:
    """Every dense score at the floor: the corpus was indexed with random
    vectors."""
    run = clean_run()
    for qr in run["question_results"]:
        for ref in qr["source_refs"]:
            ref["dense_score"] = 0.0
    return _detector_ids(run)


def bait_F17_keyword_bias() -> set[str]:
    """Almost the whole context contributed by the keyword half alone.

    Not "sparse scores are larger": they always are, being on another scale.
    """
    run = clean_run()
    for qr in run["question_results"]:
        for ref in qr["source_refs"]:
            ref["dense_score"], ref["sparse_score"] = 0.0, 9.0
    return _detector_ids(run)


def bait_F18_right_fragment_below_the_cutoff() -> set[str]:
    """Indexed, and the widened search finds it well outside the selection."""
    cause = classify_retrieval_cause(
        presences={"handbook/7": "present"},
        miss={"found": True, "rank": 34, "score": 0.4, "widened_k": 200},
        top_k=5,
        chunk_counts={"handbook/7": 1},
    )
    return {f"cause:{cause.cause}"}


def bait_F28_reasoning_model_returns_nothing() -> set[str]:
    run = clean_run()
    for qr in run["question_results"]:
        qr["generated_answer"] = ""
    return _detector_ids(run)


def bait_F29_wrong_number_in_the_citation() -> set[str]:
    """The right chunk was retrieved and the answer names another one's
    number."""
    ref = SourceRef(**{k: v for k, v in _healthy_ref(40).items()})
    wrong = citation_number_coverage("The rule is set out in section 43.", [ref], ["handbook/40"])
    right = citation_number_coverage("The rule is set out in section 40.", [ref], ["handbook/40"])
    assert right == 1.0, "the bait's own healthy case does not score as correct"
    return {"metric:citation_number_coverage"} if wrong == 0.0 else set()


def bait_F31_answer_from_parametric_knowledge() -> set[str]:
    """Nothing relevant retrieved, and a confident answer all the same."""
    verdict = diagnose_question(
        "answerable",
        {"retrieval_recall_at_k": 0.0, "answer_similarity": 0.85},
        None,
    )
    return {f"funnel:{verdict.layer}"}


def bait_F32_refusal_calibrated_badly() -> set[str]:
    run = clean_run()
    for qr in run["question_results"]:
        qr["metrics"]["correct_refusal"] = 0.0
    return _detector_ids(run)


def bait_F03_segmentation_did_nothing() -> set[str]:
    """The strategy claimed a tree and produced flat windows: no chunk carries
    a structural path."""
    chunks = [{**c, "structural_path": "root"} for c in clean_chunks()]
    return _health_ids(chunks)


def bait_F14_model_does_not_cover_the_language() -> set[str]:
    """A corpus whose documents are not all in one language."""
    chunks = clean_chunks()
    for c in chunks[:8]:
        c["text"] = ("Настоящий раздел устанавливает порядок, который применяется "
                     "в этом случае, и называет утверждающего целиком.")
    return _health_ids(chunks)


def bait_F15_semantic_search_misses_an_identifier() -> set[str]:
    """Indexed, a single chunk, and still not found at a wide k: the wording of
    the question and of the document do not meet."""
    cause = classify_retrieval_cause(
        presences={"handbook/7": "present"},
        miss={"found": False, "widened_k": 200},
        top_k=5,
        chunk_counts={"handbook/7": 1},
    )
    verdict = diagnose_question("answerable", {"retrieval_recall_at_k": 0.0, "answer_similarity": 0.2}, None)
    return {f"cause:{cause.cause}", f"funnel:{verdict.layer}"}


def bait_F21_incomparable_scales() -> set[str]:
    """The merge obeys one half, and the other reaches the context barely.

    Baited in the direction the old rule could not see, because a stored run
    has exactly this: a weighted merge where 89% of the context comes from the
    semantic half and the keyword half is paid for and idle.
    """
    run = clean_run()
    for qr in run["question_results"]:
        for ref in qr["source_refs"]:
            ref["dense_score"], ref["sparse_score"] = 0.7, 0.0
    return _detector_ids(run)


def bait_F40_one_half_is_absent() -> set[str]:
    """No chunk of any context carries a score from the second half.

    Distinct from the entry above, which has both halves present and one of
    them outweighed: here the second index was never built, so it contributes
    to nothing. Measured on the proving ground, that is 750 of 750 chunks from
    one half, against 167 of 750 when both are built.
    """
    run = clean_run()
    for qr in run["question_results"]:
        for ref in qr["source_refs"]:
            ref["dense_score"], ref["sparse_score"] = 0.7, 0.0
    return _detector_ids(run)


def bait_F16_keyword_search_misses_a_paraphrase() -> set[str]:
    return bait_F15_semantic_search_misses_an_identifier()


def bait_F24_reranker_does_not_know_the_language() -> set[str]:
    return bait_F14_model_does_not_cover_the_language()


def bait_F02_one_identifier_two_fragments() -> set[str]:
    """A source path lost at load time collapses two derivations onto one
    value, so retrieval returns the wrong text under the right identifier."""
    run = clean_run()
    for qr in run["question_results"]:
        qr["source_refs"][1]["chunk_id"] = qr["source_refs"][0]["chunk_id"]
    return _detector_ids(run)


def bait_F27_the_reranker_costs_an_unknown_amount() -> set[str]:
    """The reranker ran on every question and no question says what it cost.

    The trace is otherwise present, which is the point: a system reporting
    nothing at all is a different finding, already named elsewhere, and a
    detector that could not tell the two apart would say the wider thing
    whenever the narrower one was true.
    """
    run = clean_run()
    for qr in run["question_results"]:
        qr["stage_trace"] = {
            "embed_ms": 4.0, "dense_retrieve_ms": 20.0, "generate_ms": 900.0,
            "output_tokens": 120, "n_reranked": 5, "rerank_ms": 0.0, "total_ms": 930.0,
        }
    return _detector_ids(run)


def bait_F33_a_metric_declares_nothing() -> set[str]:
    """A number reaches the reader carrying a name and nothing else, so
    whether the name still matches what it computes cannot be asked."""
    run = clean_run()
    run["aggregate_metrics"] = {"retrieval_recall_at_k": 1.0, "answer_precision": 0.9}
    # Carried by the questions as well, so the only thing wrong with it is
    # that it declares nothing. A number in the run and on no question is a
    # different failure, and letting both stand would make this pair
    # evidence for either.
    for qr in run["question_results"]:
        qr["metrics"]["answer_precision"] = 0.9
    return _detector_ids(run)


def bait_F34_the_run_disagrees_with_its_own_questions() -> set[str]:
    """The number on the screen is not the number the questions carry: a
    question lost between writing the run and reading it moves the aggregate
    and leaves everything else looking correct."""
    run = clean_run()
    run["aggregate_metrics"] = {"retrieval_recall_at_k": 1.0}
    run["question_results"][0]["metrics"]["retrieval_recall_at_k"] = 0.0
    return _detector_ids(run)


def bait_F35_a_metric_computed_where_it_has_no_grounds() -> set[str]:
    """A recall recorded for a question the corpus does not cover.

    Nothing was retrievable, so the zero is not a measurement of retrieval;
    averaged into the run it is a confident number about something nobody
    measured.
    """
    run = clean_run()
    for qr in run["question_results"]:
        qr["answerability"] = "out_of_scope"
    return _detector_ids(run)


def _run_with(manifest: dict[str, Any], applied: dict[str, Any]) -> dict[str, Any]:
    run = clean_run()
    run["corpus_manifest"] = manifest
    run["applied"] = applied
    return run


_HEALTHY_MANIFEST = {
    "embedder_id": "bge_m3", "embedder_version": "1.0.0",
    "embedder_is_real_model": True, "document_count": 20,
    "documents_digest": "a" * 64, "loaded_at": "2026-09-01T10:00:00Z",
}
_HEALTHY_APPLIED = {
    "query_embedder_id": "bge_m3", "query_embedder_version": "1.0.0",
    "query_embedder_is_real_model": True, "index_embedder_id": "bge_m3",
}


def bait_F11_indexed_by_one_model_queried_by_another() -> set[str]:
    """A query vector from one model against vectors from another. Every
    score still arrives and describes a comparison of two coordinate
    systems that have nothing to do with each other."""
    return _detector_ids(_run_with(
        {**_HEALTHY_MANIFEST, "embedder_id": "e5_large"}, _HEALTHY_APPLIED))


def bait_F12_the_model_changed_and_the_corpus_did_not() -> set[str]:
    """The name is the same and the weights are not, so nothing about the
    mismatch appears in a collection name or in a setting."""
    return _detector_ids(_run_with(
        {**_HEALTHY_MANIFEST, "embedder_version": "0.9.0"}, _HEALTHY_APPLIED))


def bait_F09_the_corpus_changed_between_two_runs() -> set[str]:
    """One corpus name over two different sets of documents. A difference in
    the numbers is then a difference in the documents, and the platform said
    for a long time that it could not tell."""
    from core.experiment.compare import check_comparability
    from core.experiment.config import ComponentRef, ExperimentConfig
    from core.experiment.runner import ExperimentResult, QuestionResult

    def _result(digest: str, loaded: str) -> ExperimentResult:
        result = ExperimentResult(
            config=ExperimentConfig(
                name="r", corpus_id="handbook", dataset_name="handbook.v1.jsonl",
                chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
                embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
                generator=ComponentRef(kind="generator", component_id="ollama"),
            ),
            dataset_name="handbook.v1.jsonl",
        )
        result.question_results = [
            QuestionResult(question_id=f"q{i}", question="?", reference_answer="",
                           generated_answer="an answer", metrics={"retrieval_recall_at_k": 1.0})
            for i in range(1, 21)
        ]
        result.corpus_manifest = {**_HEALTHY_MANIFEST, "documents_digest": digest,
                                  "loaded_at": loaded}
        return result

    warnings = check_comparability(
        _result("a" * 64, "2026-09-01T10:00:00Z"), _result("b" * 64, "2026-09-04T10:00:00Z"))
    return {f"compare:{w.id}" for w in warnings}


def bait_F37_the_number_is_the_best_of_a_search() -> set[str]:
    """Forty configurations on one question set, and the figure reported from
    the last of them is partly the search."""
    run = clean_run()
    run["tuning_provenance"] = {
        "dataset_name": "handbook.v1.jsonl", "runs_before": 40,
        "configurations_before": 40, "fusion_constants_tried": [60], "merge_strategy": "weighted",
    }
    return _detector_ids(run)


def bait_F22_the_fusion_constant_was_never_varied() -> set[str]:
    """Every rank-fusion run on this set used the published value, so nobody
    has measured whether it is a good one here."""
    run = clean_run()
    run["tuning_provenance"] = {
        "dataset_name": "handbook.v1.jsonl", "runs_before": 0,
        "configurations_before": 0, "fusion_constants_tried": [60], "merge_strategy": "rrf",
    }
    return _detector_ids(run)


def bait_F36_a_difference_the_questions_cannot_see() -> set[str]:
    """A movement smaller than the scatter of the questions themselves,
    reported to three decimals and read as a change."""
    from core.experiment.compare import check_comparability
    from core.experiment.config import ComponentRef, ExperimentConfig
    from core.experiment.runner import ExperimentResult, QuestionResult

    scores = [0.2, 0.9, 0.5, 0.8, 0.3, 0.95, 0.4, 0.7, 0.6, 0.85]

    def _result(shift: float) -> ExperimentResult:
        moved = [round(min(1.0, score + shift), 3) for score in scores]
        result = ExperimentResult(
            config=ExperimentConfig(
                name="r", corpus_id="handbook", dataset_name="handbook.v1.jsonl",
                chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
                embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
                generator=ComponentRef(kind="generator", component_id="ollama"),
            ),
            dataset_name="handbook.v1.jsonl",
        )
        result.question_results = [
            QuestionResult(question_id=f"q{i}", question="?", reference_answer="",
                           generated_answer="an answer", metrics={"answer_similarity": score})
            for i, score in enumerate(moved, start=1)
        ]
        result.aggregate_metrics = {"answer_similarity": sum(moved) / len(moved)}
        return result

    return {f"compare:{w.id}" for w in check_comparability(_result(0.0), _result(0.01))}


BAITS = {
    "F01": bait_F01_reingest_duplicates,
    "F02": bait_F02_one_identifier_two_fragments,
    "F03": bait_F03_segmentation_did_nothing,
    "F04": bait_F04_export_lost_documents,
    "F06": bait_F06_chunks_too_small,
    "F07": bait_F07_grounds_spread_across_chunks,
    "F09": bait_F09_the_corpus_changed_between_two_runs,
    "F10": bait_F10_stub_embedder,
    "F11": bait_F11_indexed_by_one_model_queried_by_another,
    "F12": bait_F12_the_model_changed_and_the_corpus_did_not,
    "F14": bait_F14_model_does_not_cover_the_language,
    "F15": bait_F15_semantic_search_misses_an_identifier,
    "F16": bait_F16_keyword_search_misses_a_paraphrase,
    "F17": bait_F17_keyword_bias,
    "F18": bait_F18_right_fragment_below_the_cutoff,
    "F21": bait_F21_incomparable_scales,
    "F40": bait_F40_one_half_is_absent,
    "F22": bait_F22_the_fusion_constant_was_never_varied,
    "F24": bait_F24_reranker_does_not_know_the_language,
    "F27": bait_F27_the_reranker_costs_an_unknown_amount,
    "F28": bait_F28_reasoning_model_returns_nothing,
    "F29": bait_F29_wrong_number_in_the_citation,
    "F31": bait_F31_answer_from_parametric_knowledge,
    "F32": bait_F32_refusal_calibrated_badly,
    "F33": bait_F33_a_metric_declares_nothing,
    "F34": bait_F34_the_run_disagrees_with_its_own_questions,
    "F35": bait_F35_a_metric_computed_where_it_has_no_grounds,
    "F36": bait_F36_a_difference_the_questions_cannot_see,
    "F37": bait_F37_the_number_is_the_best_of_a_search,
}

_CLAIMED = [f for f in FAILURES if f.detection != "none"]


@pytest.mark.parametrize("failure", _CLAIMED, ids=lambda f: f.id)
def test_bait(failure: Any) -> None:
    """Fires on the defect, and at least one of the entry's own signals is
    among what fired."""
    reachable = [s for s in failure.signals if s.id not in STAND_ONLY]
    if not reachable:
        pytest.skip(
            f"{failure.id}: every signal it names needs the proving ground "
            f"({[STAND_ONLY[s.id] for s in failure.signals]})"
        )
    bait = BAITS.get(failure.id)
    assert bait is not None, (
        f"{failure.id} claims detection {failure.detection!r} and has no bait. "
        "An unproven claim must not pass."
    )
    fired = bait()
    expected = {s.id for s in reachable}
    assert expected & fired, (
        f"{failure.id}: none of its own signals fired on the payload carrying "
        f"the defect. Named {sorted(expected)}, fired {sorted(fired)}"
    )


@pytest.mark.parametrize("failure", _CLAIMED, ids=lambda f: f.id)
def test_bait_stays_silent_on_the_clean_baseline(failure: Any) -> None:
    """The half that a detector firing on everything would fail."""
    reachable = [s for s in failure.signals if s.id not in STAND_ONLY]
    if not reachable:
        pytest.skip(f"{failure.id}: baited on the proving ground")
    quiet = _detector_ids(clean_run()) | _health_ids(clean_chunks()) | _compare_ids()
    still_firing = {s.id for s in reachable} & quiet
    assert not still_firing, (
        f"{failure.id}: {sorted(still_firing)} fires on a healthy payload, so it "
        "cannot be evidence of anything"
    )


def test_every_claim_is_either_baited_here_or_declared_stand_only() -> None:
    """Nothing passes by omission."""
    unaccounted = []
    for f in _CLAIMED:
        if f.id in BAITS:
            continue
        if all(s.id in STAND_ONLY for s in f.signals):
            continue
        unaccounted.append(f.id)
    assert unaccounted == [], (
        f"entries claiming detection with neither a bait nor a stated reason: {unaccounted}"
    )


def test_no_fault_is_reported_on_a_healthy_dense_only_run() -> None:
    """A legitimate shape of run, and a baseline for every bait that reasons
    about dense scores.

    Faults only. An `info` saying a check could not run is not a fault, and
    saying so is the point: "not checked" must be distinguishable from
    "checked and fine", and from "checked and broken".
    """
    faults = [d.id for d in run_detectors(clean_dense_only_run()) if d.severity in ("warn", "error")]
    assert faults == [], (
        "a healthy dense-only run is reported as faulty. Every one of the "
        f"stored dense runs on this machine carries this: {faults}"
    )


def test_the_platform_says_when_it_could_not_check_the_embedder() -> None:
    """The other half of the previous test, and the reason it is not simply a
    silent skip.

    On a dense-only run nothing records a per-signal score split, so whether
    the corpus was indexed with a real model cannot be established from what
    the run stored. Staying silent would let a reader take the absence of a
    warning for a clean bill of health.
    """
    reported = {d.id: d.severity for d in run_detectors(clean_dense_only_run())}
    assert "embedder_unverified" in reported, (
        "a dense-only run says nothing about whether its embedder was real, "
        f"and does not say that it says nothing: {reported}"
    )
    assert reported["embedder_unverified"] == "info"


def test_the_embedder_check_still_speaks_where_it_can() -> None:
    """The fix must not turn a false alarm into a permanent silence: on a run
    that does record a score split, a collapsed dense side is still a fault."""
    run = clean_run()
    for qr in run["question_results"]:
        for ref in qr["source_refs"]:
            ref["dense_score"] = 0.0        # split recorded, dense side at the floor
            ref["sparse_score"] = 4.0
    fired = [d.id for d in run_detectors(run)]
    assert "stub_embedder" in fired, f"a genuinely collapsed dense side is no longer caught: {fired}"


def test_no_detector_fires_on_a_healthy_retrieval_only_run() -> None:
    """A run that stopped before the generator answers nothing by design."""
    fired = [d.id for d in run_detectors(clean_retrieval_only_run())]
    assert fired == [], (
        "a run that deliberately stopped before generation is reported as "
        f"faulty for having no answers: {fired}"
    )
