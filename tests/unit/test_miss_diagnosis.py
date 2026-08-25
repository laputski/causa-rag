"""core/eval/miss_diagnosis.py — re-query with a wider k to see how close
retrieval actually came on a recall_at_k=0 question."""
from __future__ import annotations

from unittest.mock import MagicMock

from core.eval.miss_diagnosis import diagnose_retrieval_miss
from core.models import Chunk, ScoredChunk


def _scored(source_code: str, article_no: str, score: float, path: str = "") -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            doc_id="d1", text="x", structural_path=path,
            metadata={"source_code": source_code, "article_no": article_no},
        ),
        score=score,
    )


def _fake_pipeline(retrieved: list[ScoredChunk]) -> MagicMock:
    pipeline = MagicMock()
    pipeline._embedder.embed.return_value = [[0.1, 0.2, 0.3]]
    pipeline._retriever.retrieve.return_value = retrieved
    return pipeline


def test_found_reports_rank_and_score() -> None:
    retrieved = [
        _scored("SRC001", "44", 0.9, path="document/article[Article 44]"),
        _scored("SRC001", "210", 0.7, path="document/article[Article 210]"),
        _scored("SRC001", "5", 0.5, path="document/article[Article 5]"),
    ]
    pipeline = _fake_pipeline(retrieved)

    out = diagnose_retrieval_miss(pipeline, "a question", ["SRC001/5"], widened_k=50)

    assert out == {
        "found": True, "rank": 3, "score": 0.5,
        "structural_path": "document/article[Article 5]", "widened_k": 50,
    }


def test_not_found_even_at_widened_k() -> None:
    retrieved = [_scored("SRC001", "44", 0.9), _scored("SRC001", "210", 0.7)]
    pipeline = _fake_pipeline(retrieved)

    out = diagnose_retrieval_miss(pipeline, "a question", ["SRC001/999"], widened_k=50)

    assert out == {"found": False, "rank": None, "score": None, "structural_path": None, "widened_k": 50}


def test_widened_k_is_passed_through_to_retriever() -> None:
    pipeline = _fake_pipeline([])
    diagnose_retrieval_miss(pipeline, "a question", ["X/1"], widened_k=37)
    _, kwargs = pipeline._retriever.retrieve.call_args
    assert kwargs["k"] == 37


def test_multiple_article_refs_matches_best_rank() -> None:
    """Two acceptable ground-truth refs — the closer one's rank is reported."""
    retrieved = [
        _scored("A", "1", 0.9),
        _scored("B", "2", 0.8),  # this one matches and ranks first among matches
        _scored("C", "3", 0.6),
        _scored("B", "9", 0.4),  # this one ALSO matches (B/9 not in refs) -- not a match actually
    ]
    pipeline = _fake_pipeline(retrieved)
    out = diagnose_retrieval_miss(pipeline, "a question", ["B/2", "C/3"], widened_k=10)
    assert out["found"] is True
    assert out["rank"] == 2
    assert out["score"] == 0.8
