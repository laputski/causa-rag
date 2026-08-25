"""The wiring — the cause reaches the reader, and the widened re-query
recognises every form of ref id.

Two things are pinned here that the pure unit tests cannot reach: a computed
verdict surviving the trip to storage and back (a field being dropped between
writer and reader has already happened twice in this codebase), and the
ref-id matching inside the widened re-query.
"""
from __future__ import annotations

from core.eval.miss_diagnosis import diagnose_retrieval_miss
from core.experiment.config import ComponentRef, ExperimentConfig
from core.experiment.runner import ExperimentResult, QuestionResult
from core.models import Chunk, ScoredChunk
from services.api_gateway.routers.experiments import _diagnose_root_causes, _parse_result


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        name="run", top_k=5,
        chunking_strategy=ComponentRef(kind="chunker", component_id="structure_aware"),
        embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
        generator=ComponentRef(kind="generator", component_id="ollama"),
    )


# ── The verdict survives the round trip ─────────────────────────────────

def test_root_cause_survives_serialisation_and_reading_back() -> None:
    result = ExperimentResult(config=_config())
    result.question_results.append(QuestionResult(
        question_id="q1", question="q", reference_answer="r", generated_answer="g",
        root_cause={"cause": "data_missing", "detail": "not in the index", "evidence": {}},
    ))
    parsed = _parse_result(result.to_dict())
    assert parsed is not None
    assert parsed.question_results[0].root_cause["cause"] == "data_missing"


def test_a_run_stored_before_this_field_existed_reads_back_as_none() -> None:
    data = ExperimentResult(config=_config()).to_dict()
    data["question_results"] = [{"question_id": "q1", "question": "q"}]
    assert _parse_result(data).question_results[0].root_cause is None


# ── Only an ambiguous verdict is analysed ───────────────────────────────

class _Resolver:
    checked = True

    def __init__(self, known: set[str]) -> None:
        self._known = known

    def presence(self, ref: str) -> str:
        return "present" if ref in self._known else "absent"


def _result_with(metrics: dict) -> ExperimentResult:
    result = ExperimentResult(config=_config())
    result.question_results.append(QuestionResult(
        question_id="q1", question="q", reference_answer="r", generated_answer="g",
        metrics=metrics, answerability="answerable",
    ))
    return result


def test_a_retrieval_failure_gets_a_cause() -> None:
    result = _result_with({"retrieval_recall_at_k": 0.0, "answer_similarity": 0.1})
    _diagnose_root_causes(
        result, None, _Resolver(set()), {"q1": ["S/1"]},
    )
    assert result.question_results[0].root_cause["cause"] == "data_missing"


def test_a_healthy_question_is_not_analysed_at_all() -> None:
    """The widened re-query costs a retrieval per question it runs on. It
    must run on failures only, or a large healthy run pays for nothing."""
    result = _result_with({"retrieval_recall_at_k": 1.0, "answer_similarity": 0.9,
                           "grounded_in_correct_source": 0.9})
    _diagnose_root_causes(result, None, _Resolver({"S/1"}), {"q1": ["S/1"]})
    assert result.question_results[0].root_cause is None


def test_a_rerank_failure_keeps_its_own_verdict_rather_than_being_reclassified() -> None:
    result = _result_with({
        "retrieval_recall_at_k": 0.0, "answer_similarity": 0.1,
        "pre_rerank_recall_at_k": 1.0,
    })
    _diagnose_root_causes(result, None, _Resolver({"S/1"}), {"q1": ["S/1"]})
    assert result.question_results[0].root_cause is None


# ── The widened re-query understands all three ref-id forms ─────────────

class _StubPipeline:
    """Returns one chunk that carries no source_code at all — the general
    document corpus shape, where a ref id is `doc_id#structural_path`."""

    class _Embedder:
        def embed(self, texts):
            return [[0.0, 1.0] for _ in texts]

    class _Retriever:
        def retrieve(self, query, k, filters=None, query_vector=None):
            return [ScoredChunk(
                chunk=Chunk(doc_id="D1", text="t", structural_path="Ch 2 > Art 10"),
                score=0.7, retriever_id="stub",
            )]

    def __init__(self) -> None:
        self._embedder = self._Embedder()
        self._retriever = self._Retriever()


def test_the_widened_query_matches_a_structural_path_ref_not_only_a_numbered_one() -> None:
    """Was `f"{source_code}/{article_no}"`, the first of three forms only.
    Any corpus not laid out as one numbered file per citable unit reported
    "not found even at a wide k" for a chunk sitting at rank 1, which reads
    as an indexing problem when nothing is wrong."""
    found = diagnose_retrieval_miss(_StubPipeline(), "a question", ["D1#Ch 2 > Art 10"], widened_k=50)
    assert found["found"] is True
    assert found["rank"] == 1


def test_a_ref_that_is_genuinely_elsewhere_still_reports_not_found() -> None:
    missing = diagnose_retrieval_miss(_StubPipeline(), "a question", ["D9#no such thing"], widened_k=50)
    assert missing["found"] is False
    assert missing["widened_k"] == 50


# ── Found live: an id-keyed lookup matched nothing ──────────────────────

class _Dataset:
    def __init__(self, questions):
        self.questions = questions


def test_refs_are_found_for_a_dataset_whose_questions_have_no_ids() -> None:
    """A dataset authored before per-question ids existed stores every
    question with `id: None`. An id-keyed lookup collapsed them onto one
    entry and matched none, so every retrieval failure reported "no expected
    refs" — a confident wrong verdict, which is the worst kind because it
    reads as an answer. Caught by running a real experiment, not by a test.
    """
    from services.api_gateway.routers.experiments import _expected_refs_index

    index = _expected_refs_index(_Dataset([
        {"id": None, "question": "How many axes?", "article_refs": ["RC/1"]},
        {"id": None, "question": "What is hybrid retrieval?", "article_refs": ["RC/2"]},
    ]))
    assert index["How many axes?"] == ["RC/1"]
    assert index["What is hybrid retrieval?"] == ["RC/2"]


def test_an_id_is_preferred_over_text_when_present() -> None:
    from services.api_gateway.routers.experiments import _expected_refs_index

    index = _expected_refs_index(_Dataset([
        {"id": "q1", "question": "a question", "article_refs": ["S/1"]},
    ]))
    assert index["q1"] == ["S/1"]
    assert index["a question"] == ["S/1"]


def test_the_diagnosis_falls_back_to_question_text() -> None:
    result = _result_with({"retrieval_recall_at_k": 0.0, "answer_similarity": 0.1})
    result.question_results[0].question_id = ""
    result.question_results[0].question = "How many axes?"
    _diagnose_root_causes(
        result, None, _Resolver(set()), {"How many axes?": ["RC/1"]},
    )
    assert result.question_results[0].root_cause["cause"] == "data_missing"
