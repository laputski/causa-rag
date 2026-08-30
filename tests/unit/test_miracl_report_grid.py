"""eval/miracl/report.py — the grid's own settings, checked without infrastructure.

The run itself needs Qdrant, OpenSearch and 25 GB of weights. What can be
held here is everything that decides what the run measures: that no row
generates, that the reranked and unreranked halves see the same candidate
window, and that a metric nobody measured reaches the published table as a
gap rather than as a zero.
"""
from __future__ import annotations

import pytest

from eval.miracl import report


def test_every_config_is_retrieval_only():
    for mode in report.MODES:
        for k in report.TOP_K:
            for rr in (False, True):
                cfg = report._config("ar", mode, k, rr, "bge_m3")
                assert cfg.retrieval_only is True, cfg.name


def test_the_generator_slot_refuses_to_generate():
    """The report's claim is that it never generated. A quiet stub would
    make the claim unfalsifiable."""
    with pytest.raises(AssertionError, match="retrieval-only report"):
        report._RefusingGenerator().generate("any prompt")


def test_candidate_window_is_constant_across_the_grid():
    """The reranker's effect is only readable while the two halves are
    given the same fifty passages. A fetch_k that followed top_k would move
    the window and the reranker together."""
    windows = {
        report._config("ar", mode, k, rr, "bge_m3").fetch_k
        for mode in report.MODES for k in report.TOP_K for rr in (False, True)
    }
    assert windows == {report.FETCH_K}
    assert max(report.TOP_K) < report.FETCH_K


def test_reranker_is_attached_only_when_asked():
    assert report._config("ar", "naive", 5, False, "bge_m3").reranker is None
    assert report._config("ar", "naive", 5, True, "bge_m3").reranker.component_id == report.RERANKER


def test_corpus_id_matches_the_fetcher_layout():
    assert report._config("ru", "naive", 10, False, "bge_m3").corpus_id == "miracl-ru"


def test_config_names_are_unique_across_the_grid():
    """They become filenames and table rows; a collision would silently
    overwrite one measurement with another."""
    names = [
        report._config(lang, mode, k, rr, "bge_m3").name
        for lang in report.LANGUAGES for mode in report.MODES
        for k in report.TOP_K for rr in (False, True)
    ]
    assert len(names) == len(set(names))


def _row(metrics: dict[str, float]) -> report.Row:
    return report.Row(
        language="ar", mode="naive", top_k=5, reranker=False, reranker_model="",
        n_questions=200, n_failed=0, seconds=1.0, metrics=metrics,
    )


def test_an_unmeasured_metric_is_not_printed_as_zero():
    """The same rule the evaluator now follows, carried into the table. A
    dash reads as 'we did not measure this'; 0.000 reads as a result."""
    md = report.to_markdown([_row({"retrieval_recall_at_k": 0.5})])
    assert "0.500" in md
    assert "not measured" in md
    assert "0.000" not in md


def test_a_measured_zero_is_still_printed():
    """The bait. Retrieval that genuinely found nothing must not be
    disguised as an absent measurement."""
    md = report.to_markdown([_row({
        "retrieval_recall_at_k": 0.0,
        "retrieval_precision_at_k": 0.0,
        "retrieval_average_precision": 0.0,
    })])
    assert "0.000" in md
    assert "not measured" not in md


def test_the_reranker_default_is_multilingual():
    """The platform's own local cross-encoder is English-only. Two of the
    three languages in this report would have been scored on noise."""
    assert "m3" in report.RERANKER_MODEL
    assert "ms-marco" not in report.RERANKER_MODEL


def test_chunk_size_clears_the_passage_length():
    """Above the 99th percentile in all three languages, so the chunk-size
    axis stays out of the grid instead of confounding it."""
    assert report.CHUNK_SIZE > 2553
    assert report.CHUNK_OVERLAP == 0


def test_the_document_leads_with_the_caveats():
    """The plan's own condition. A reader who meets the shortened pool in a
    footnote has already read the table as if it were the whole corpus."""
    md = report.to_document([_row({"retrieval_recall_at_k": 0.5})], limit=None)
    caveats = md.index("Read these three things first")
    results = md.index("## Results")
    assert caveats < results
    for phrase in ("judged passages", "dev split", "No generation is measured"):
        assert phrase in md


def test_the_document_records_which_reranker_ranked_the_rows():
    row = report.Row(
        language="ar", mode="naive", top_k=5, reranker=True,
        reranker_model="BAAI/bge-reranker-v2-m3", n_questions=10, n_failed=0,
        seconds=1.0, metrics={"retrieval_recall_at_k": 0.4},
    )
    assert "BAAI/bge-reranker-v2-m3" in report.to_document([row], limit=None)


def test_the_document_says_when_a_run_was_limited():
    """A smoke run's numbers must not be readable as the full dev split."""
    md = report.to_document([_row({"retrieval_recall_at_k": 0.5})], limit=200)
    assert "first 200 questions" in md
    assert "every question in the dev split" not in md


def test_failed_questions_are_counted_in_the_table():
    """A row averaged over the tenth of its questions that succeeded is not
    a measurement of the other nine, and the table has to say so."""
    row = report.Row(
        language="ar", mode="naive", top_k=5, reranker=False, reranker_model="",
        n_questions=20, n_failed=180, seconds=1.0,
        metrics={"retrieval_recall_at_k": 0.9},
    )
    md = report.to_markdown([row])
    assert "| 20 | 180 |" in md


class _CountingReranker:
    """Reverses the candidates and counts how often it was actually asked."""

    reranker_id = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, query, candidates):
        from core.models import ScoredChunk
        self.calls += 1
        return [
            ScoredChunk(chunk=c.chunk, score=float(i), retriever_id=self.reranker_id)
            for i, c in enumerate(reversed(candidates))
        ]


def _cands(n: int):
    from core.models import Chunk, ScoredChunk
    return [
        ScoredChunk(chunk=Chunk(doc_id="d", chunk_id=f"c{i}", text=f"text {i}"),
                    score=1.0, retriever_id="x")
        for i in range(n)
    ]


def test_memoised_reranker_returns_exactly_what_the_inner_one_did():
    """The saving is only legitimate while the answer is identical, scores
    and all — the pipeline cuts by position and the evaluator reads score."""
    inner = _CountingReranker()
    memo = report._MemoisingReranker(inner)
    cands = _cands(5)

    first = memo.rerank("q", cands)
    second = memo.rerank("q", cands)

    assert [c.chunk.chunk_id for c in first] == [c.chunk.chunk_id for c in second]
    assert [c.score for c in first] == [c.score for c in second]
    assert [c.chunk.chunk_id for c in first] == ["c4", "c3", "c2", "c1", "c0"]


def test_the_same_question_is_scored_once_across_the_top_k_rows():
    inner = _CountingReranker()
    memo = report._MemoisingReranker(inner)
    cands = _cands(50)

    for _ in range(3):  # top-k 5, 10 and 20 ask the identical question
        memo.rerank("q", cands)

    assert inner.calls == 1
    assert (memo.misses, memo.hits) == (1, 2)


def test_a_different_question_is_scored_again():
    """The bait. A cache keyed too loosely would hand one question's
    ordering to another and the table would never show it."""
    inner = _CountingReranker()
    memo = report._MemoisingReranker(inner)
    cands = _cands(5)

    memo.rerank("first question", cands)
    memo.rerank("second question", cands)

    assert inner.calls == 2


def test_a_different_candidate_set_is_scored_again():
    inner = _CountingReranker()
    memo = report._MemoisingReranker(inner)

    memo.rerank("q", _cands(5))
    memo.rerank("q", _cands(6))

    assert inner.calls == 2


def test_cached_scores_survive_a_restart(tmp_path):
    """The point of the cache. An Arabic rerank takes about four seconds on
    real passages, so an interrupted language is hours of work, and a cache
    that lives only in the process means "resume" loses all of it."""
    inner = _CountingReranker()
    path = tmp_path / "rerank-ar.json"
    cands = _cands(10)

    first = report._MemoisingReranker(inner, model_name="m", cache_path=path)
    before = first.rerank("q", cands)
    first.save()

    second_inner = _CountingReranker()
    second = report._MemoisingReranker(second_inner, model_name="m", cache_path=path)
    after = second.rerank("q", cands)

    assert second_inner.calls == 0
    assert second.loaded == 1
    assert [c.chunk.chunk_id for c in after] == [c.chunk.chunk_id for c in before]
    assert [c.score for c in after] == [c.score for c in before]


def test_changing_the_reranker_invalidates_the_cache(tmp_path):
    """The bait that matters most. Replaying one model's ranking under
    another model's name would publish a row about a reranker that never
    saw the data."""
    path = tmp_path / "rerank-ar.json"
    cands = _cands(10)

    first = report._MemoisingReranker(_CountingReranker(), model_name="ms-marco", cache_path=path)
    first.rerank("q", cands)
    first.save()

    same_inner = _CountingReranker()
    same = report._MemoisingReranker(same_inner, model_name="ms-marco", cache_path=path)
    same.rerank("q", cands)
    assert same_inner.calls == 0, "the same model should read its own cached scores"

    other_inner = _CountingReranker()
    other = report._MemoisingReranker(other_inner, model_name="bge-m3", cache_path=path)
    other.rerank("q", cands)
    assert other_inner.calls == 1, "a different model must not inherit the old one's ranking"


def test_a_missing_cache_file_is_not_an_error(tmp_path):
    memo = report._MemoisingReranker(_CountingReranker(), model_name="m",
                                     cache_path=tmp_path / "absent.json")
    assert memo.loaded == 0
    assert memo.rerank("q", _cands(3))


def test_the_cache_is_written_atomically(tmp_path):
    path = tmp_path / "rerank-ar.json"
    memo = report._MemoisingReranker(_CountingReranker(), model_name="m", cache_path=path)
    memo.rerank("q", _cands(3))
    memo.save()
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_the_reading_is_absent_rather_than_empty(tmp_path, monkeypatch):
    """A heading with nothing under it reads as a finished thought that
    says nothing, which is worse than saying nothing at all."""
    monkeypatch.setattr(report, "READING", tmp_path / "absent.md")
    md = report.to_document([_row({"retrieval_recall_at_k": 0.5})], limit=None)
    assert "## Reading" not in md


def test_a_written_reading_reaches_the_document(tmp_path, monkeypatch):
    path = tmp_path / "reading.md"
    path.write_text("Dense ranks better; hybrid finds more.", encoding="utf-8")
    monkeypatch.setattr(report, "READING", path)

    md = report.to_document([_row({"retrieval_recall_at_k": 0.5})], limit=None)

    assert "## Reading" in md
    assert "Dense ranks better; hybrid finds more." in md
    # Before the reproduction instructions, after the table it reads.
    assert md.index("## Results") < md.index("## Reading") < md.index("## Reproducing it")


def test_a_reading_file_of_whitespace_is_treated_as_unwritten(tmp_path, monkeypatch):
    path = tmp_path / "reading.md"
    path.write_text("\n\n   \n", encoding="utf-8")
    monkeypatch.setattr(report, "READING", path)
    assert "## Reading" not in report.to_document([_row({"retrieval_recall_at_k": 0.5})], limit=None)


def test_the_document_names_the_cells_it_does_not_have():
    """A wrong row is guarded everywhere in this file. A missing one is
    invisible, and "Arabic was never reranked" reads exactly like "Arabic's
    reranked row is absent because reranking did not help"."""
    rows = [_row({"retrieval_recall_at_k": 0.5})]  # ar / naive / k=5 / no reranker
    md = report.to_document(rows, limit=None)

    assert "Not measured yet" in md
    assert "ar naive k=5 reranker on" in md
    assert "ar hybrid_rrf k=20 reranker off" in md


def test_a_complete_grid_says_nothing_about_missing_cells():
    """The bait. A notice that always appears stops being read."""
    rows = [
        report.Row("ar", mode, k, rr, "m" if rr else "", 10, 0, 1.0,
                   {"retrieval_recall_at_k": 0.5})
        for mode in report.MODES for k in report.TOP_K for rr in (False, True)
    ]
    assert "Not measured yet" not in report.to_document(rows, limit=None)


def test_an_unreachable_qdrant_is_not_reported_as_an_empty_corpus(monkeypatch):
    """Three states, not two. A guard that folds "the server is down" into
    "nothing is indexed" sends the reader to re-run an ingest that was
    never the problem, and the ingest fails the same way."""
    import qdrant_client

    class _Refusing:
        def __init__(self, **kw):
            raise ConnectionError("connection refused")

    monkeypatch.setattr(qdrant_client, "QdrantClient", _Refusing)
    assert report.indexed_count("en", "bge_m3") is None


def test_a_missing_collection_is_still_zero(monkeypatch):
    """The bait. Widening the guard must not swallow the case it was
    written for."""
    import qdrant_client

    class _Empty:
        def __init__(self, **kw): pass
        def collection_exists(self, name): return False

    monkeypatch.setattr(qdrant_client, "QdrantClient", _Empty)
    assert report.indexed_count("en", "bge_m3") == 0
