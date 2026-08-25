"""A toy "RAG" with no relation to this platform's own pipeline classes —
just two plain functions, the way any real external RAG would look from
the platform's point of view. Stands in for "your actual retrieval/
generation code" in this quickstart.
"""
from __future__ import annotations

from typing import Any

_CORPUS: dict[str, dict[str, str]] = {
    "doc1": {"source_code": "DEMO", "article_no": "1", "text": "Annual leave is at least 24 calendar days per year."},
    "doc2": {"source_code": "DEMO", "article_no": "2", "text": "The working week must not exceed 40 hours."},
}


def retrieve(query: str, top_k: int) -> list[dict[str, Any]]:
    """Pretend full-text search: a real RAG would call its own
    vector/keyword index here. Returns the platform's tier-1 SourceRef
    shape directly — doc_id/source_code/article_no are what
    core/eval/retrieval_metrics.py needs to score recall/precision.
    """
    query_lower = query.lower()
    hits = []
    for doc_id, doc in _CORPUS.items():
        if any(word in doc["text"].lower() for word in query_lower.split()):
            hits.append({
                "doc_id": doc_id,
                "chunk_text": doc["text"],
                "source_code": doc["source_code"],
                "article_no": doc["article_no"],
                "structural_path": f"[{doc['article_no']}]",
            })
    return hits[:top_k]


def generate(query: str, sources: list[dict[str, Any]]) -> str:
    """Pretend LLM call: a real RAG would prompt its own model here."""
    if not sources:
        return "No answer found in the available material."
    citations = ", ".join(f"Article {s['article_no']}" for s in sources)
    return f"On {query!r}: see {citations}. {sources[0]['chunk_text']}"
