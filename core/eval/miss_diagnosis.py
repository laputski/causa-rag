""""How close did retrieval actually come" for a recall_at_k=0 question
(Eval Measurement Trustworthiness, deferred item from Phase 1's plan: "needs
expanding fetch_k and storing a wider candidate pool" — implemented here as
an on-demand re-query with a wider k, not a change to every live pipeline
run, since the wider pool is only useful for this one diagnostic question).

retrieval_recall_at_k=0 alone doesn't distinguish two very different
failures: the correct chunk ranked just outside the configured top_k (a
tuning problem — raise top_k or improve ranking), or it's nowhere near the
top even at a much wider k (an indexing/embedding problem — the chunk isn't
semantically close to the query at all). Re-running retrieval with a wider
k and checking where the ground-truth chunk(s) actually land answers that.
"""
from __future__ import annotations

from typing import Any

from core.models import SourceRef


def diagnose_retrieval_miss(
    pipeline: Any, question_text: str, article_refs: list[str], widened_k: int = 50,
) -> dict[str, Any]:
    """Re-runs the SAME retriever/embedder the pipeline already uses
    (pipeline._retriever/_embedder — same private-attribute access pattern
    already used by services/api_gateway/routers/corpus.py's diagnostics
    endpoints) with a wider k, and reports the best rank/score among the
    chunks matching article_refs.

    Returns a dict with ``found`` (bool), ``rank`` (1-based position in the
    widened list, or None), ``score`` (the matching chunk's retrieval
    score, or None), ``structural_path`` (for display), and ``widened_k``
    (echoed back so the caller/UI knows what was actually searched).
    """
    from core.eval.retrieval_metrics import extract_ref_id
    from core.pipeline import _to_source_refs  # local import — avoids a cycle at module load

    query_vec = pipeline._embedder.embed([question_text])[0]
    scored = pipeline._retriever.retrieve(
        query=question_text, k=widened_k, filters=None, query_vector=query_vec,
    )
    refs: list[SourceRef] = _to_source_refs(scored)

    expected = set(article_refs)
    for rank, ref in enumerate(refs, start=1):
        # Was `f"{ref.source_code}/{ref.article_no}"`, which is
        # only the first of extract_ref_id's three forms. Any corpus not laid
        # out as one numbered file per citable unit produced "not found even
        # at a wide k" for a chunk sitting at rank 1, and that reads as an
        # indexing problem when nothing is wrong at all. Same coupling class
        # answerability shed the same coupling earlier; this was the last
        # caller still building a ref id by hand.
        if extract_ref_id(ref.model_dump()) in expected:
            return {
                "found": True,
                "rank": rank,
                "score": ref.score,
                "structural_path": ref.structural_path,
                "widened_k": widened_k,
            }
    return {"found": False, "rank": None, "score": None, "structural_path": None, "widened_k": widened_k}
