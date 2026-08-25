"""Retrieval pins overlay — the instance-level exception mechanism.

A "pin" is a small, human-authored correction tied to one
reviewer-reported question: for any future query whose meaning is close
enough to that question (measured as cosine similarity between the two
questions' embeddings), pull specific chunks into the answer's context
(``pin``), remove specific chunks from it (``demote``), or fold in the
results of an additional, differently-worded query (``rewrite``).

This module is pure logic only. It performs no MongoDB reads, no HTTP
calls, and no embedding computation of its own — the caller supplies an
already-computed query embedding and an already-fetched list of ``Pin``
records (``services/api_gateway/routers/judgments.py`` owns the
collection and the embedding call; ``core/pipeline.py`` owns invoking this
module inside a live query). Keeping it pure is what makes it unit
testable on plain fixtures and keeps ``core/`` free of the adapter/service
dependencies the layer-isolation fitness function forbids (see
the design notes).

Where this runs relative to reranking is a deliberate, temporary
compromise, not an oversight. The design this module was written for
injects a pin between the retrieval merge step and the reranker, so an
injected chunk is still evaluated (and could still be dropped) by
reranking like any other candidate. That placement depends on the
reranker seeing a wider candidate window than ``top_k``, which has not
shipped yet: today, the reranker only ever sees
exactly ``top_k`` candidates and cuts to ``top_k`` again, so a pin
injected before reranking would very likely be immediately discarded by
that same cut. ``core/pipeline.py`` therefore calls this module AFTER any
configured reranker has already run, right before the final context is
assembled — a pin-injected chunk currently enters the answer's context
unconditionally, without being evaluated by the reranker. This is less
correct than the target design, but it is functional, deterministic, and
fully visible in the run's trace (``SourceRef.pinned``), rather than
silently doing nothing. The injection point should move once F1 ships.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.models import Chunk, ScoredChunk

# Standard Reciprocal Rank Fusion constant — mirrors
# core/retrieval/hybrid.py's own default (`_rrf_score`'s `k=60`) so a
# rewrite's extra candidates are folded in with the same fusion behavior
# the platform's own dense+sparse merge already uses, not a second,
# differently-tuned formula.
_RRF_K = 60

# Added to the current top score for each successively injected pin chunk,
# so every pinned chunk sorts above every naturally-retrieved chunk without
# colliding on an identical score (which would make their relative order
# among themselves arbitrary/unstable).
_PIN_SCORE_EPSILON = 0.001


@dataclass
class Pin:
    """One ``retrieval_pins`` Mongo document, already resolved into a plain
    value object. The Mongo document itself (owned by
    ``services/api_gateway/routers/judgments.py``) is a superset of this shape —
    ``realm_id``, ``corpus_id``, ``created_by``, ``ttl_days``, ``status``,
    and so on — none of which this module has any reason to know about;
    only what actually affects one query's retrieval result is represented
    here.

    ``pin_chunks`` are a snapshot of the chunk text and structural path
    taken at the moment the pin was created (not a live corpus lookup by
    ``chunk_id`` at query time) — the router captures this snapshot
    once, so a pin keeps working even if the corpus is later re-indexed
    and the same ``chunk_id`` comes to mean something else or stops
    existing (a known, accepted limitation of the current, position-derived
    ``chunk_id`` scheme that this module does not attempt to solve).
    """

    id: str
    query_vec: list[float]
    threshold: float
    pin_chunks: list[Chunk] = field(default_factory=list)
    demote_chunk_ids: list[str] = field(default_factory=list)
    rewrite: str | None = None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Standard cosine similarity. Returns 0.0 (never raises) for empty,
    mismatched-length, or zero-magnitude vectors — an honest "does not
    match" rather than a crash on a malformed embedding."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def select_matching_pins(query_vec: list[float], pins: list[Pin]) -> list[Pin]:
    """Every pin whose own threshold the current query's embedding clears —
    a query can legitimately match more than one pin at once (two reviewer
    reports whose questions happen to be similar to each other); each
    matching pin is folded into the result independently by
    ``apply_overlay``, not just the single closest one."""
    return [p for p in pins if cosine_similarity(query_vec, p.query_vec) >= p.threshold]


def _rrf_merge(primary: list[ScoredChunk], extra: list[ScoredChunk]) -> list[ScoredChunk]:
    """Reciprocal Rank Fusion of two already-ranked chunk lists, keyed by
    ``chunk_id`` — the same formula and constant `core/retrieval/hybrid.py`
    already uses to merge dense and sparse results, applied here to merge a
    pin's ``rewrite`` query's own results with the original candidates."""
    scores: dict[str, float] = {}
    by_id: dict[str, ScoredChunk] = {}
    for ranked_list in (primary, extra):
        for rank, sc in enumerate(ranked_list, start=1):
            cid = sc.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
            if cid not in by_id:
                by_id[cid] = sc
    merged = [
        ScoredChunk(chunk=by_id[cid].chunk, score=score, retriever_id=by_id[cid].retriever_id)
        for cid, score in scores.items()
    ]
    return sorted(merged, key=lambda sc: sc.score, reverse=True)


def apply_overlay(
    scored: list[ScoredChunk],
    matching_pins: list[Pin],
    rewrite_results: dict[str, list[ScoredChunk]] | None = None,
) -> list[ScoredChunk]:
    """Folds every matching pin's instructions into ``scored``, in a fixed
    order: demote first, then pin, then rewrite.

    Demoting removes a chunk from the candidate list entirely (an explicit
    exclusion, not a soft score penalty — simpler to reason about and to
    test: a demoted chunk is either present or it is not, never "present
    but suppressed by some unspecified amount"). Pinning injects each
    listed chunk — skipping any whose ``chunk_id`` is already present, so a
    pin never creates a duplicate of a chunk retrieval already found — at a
    score placed just above the current top score, so every pinned chunk
    outranks every naturally-retrieved one. A pin whose ``rewrite`` is set
    is RRF-merged with ``rewrite_results[pin.id]`` when the caller supplied
    it; a pin with ``rewrite`` set but no corresponding entry in
    ``rewrite_results`` is treated exactly as if ``rewrite`` were absent
    (the caller may simply not have attempted the extra retrieval call —
    honest degradation, not a crash).

    Returns a new list, re-sorted by score descending; ``scored`` itself is
    never mutated.
    """
    if not matching_pins:
        return list(scored)

    demote_ids: set[str] = set()
    for pin in matching_pins:
        demote_ids.update(pin.demote_chunk_ids)
    result = [sc for sc in scored if sc.chunk.chunk_id not in demote_ids]

    existing_ids = {sc.chunk.chunk_id for sc in result}
    top_score = max((sc.score for sc in result), default=0.0)
    for pin in matching_pins:
        for chunk in pin.pin_chunks:
            if chunk.chunk_id in existing_ids:
                continue
            top_score += _PIN_SCORE_EPSILON
            tagged = chunk.model_copy(update={
                "metadata": {**chunk.metadata, "pinned": True, "pin_id": pin.id},
            })
            result.append(ScoredChunk(chunk=tagged, score=top_score, retriever_id="pin_overlay"))
            existing_ids.add(chunk.chunk_id)

    for pin in matching_pins:
        if not pin.rewrite:
            continue
        extra = (rewrite_results or {}).get(pin.id)
        if not extra:
            continue
        result = _rrf_merge(result, extra)

    return sorted(result, key=lambda sc: sc.score, reverse=True)
