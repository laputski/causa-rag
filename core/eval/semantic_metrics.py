"""Semantic answer-quality metrics (Eval Measurement Trustworthiness, Phase
0) — replaces the Jaccard token-overlap metrics in
services/api_gateway/routers/experiments.py:_BasicEvaluator, which were
structurally incapable of producing a meaningful faithfulness score (see
core/eval/answerability.py module docstring and the Phase 0 plan for why).

Cosine similarity over embeddings, not token sets — robust to Russian
morphology (paraphrase, declension) that defeats word-set overlap.

Takes an embedder explicitly (any object with .embed(list[str]) ->
list[list[float]], e.g. adapters.bge_m3.BgeM3Embedder) rather than
constructing one internally — keeps this module deterministic and testable
without a GPU/model load, and lets callers reuse the same embedder instance
(and its cache) already in the pipeline.
"""
from __future__ import annotations

import math
from typing import Any, Protocol


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def answer_similarity(answer: str, reference: str, embedder: Embedder) -> float:
    """Cosine similarity between the generated answer and the reference
    (ground_truth) answer. 1.0 = semantically identical, not a paraphrase
    detector at the word level — robust to rewording, declension, length.
    """
    if not answer.strip() or not reference.strip():
        return 0.0
    vecs = embedder.embed([answer, reference])
    return _cosine(vecs[0], vecs[1])


def answer_relevance(answer: str, question: str, embedder: Embedder) -> float:
    """Cosine similarity between the generated answer and the QUESTION it was
    asked — independent of whether the answer is factually correct.

    Deterministic counterpart to deepeval's AnswerRelevancyMetric (LLM judge):
    answer_similarity/context_support/grounded_in_correct_source all compare
    the answer against ground_truth or retrieved context, never against the
    question itself — a confidently on-topic-sounding but wrong answer and a
    correct-but-tangential one both score fine there. This catches the case
    isolated metrics don't: did the model actually engage with what was
    asked, regardless of whether it got the content right.
    """
    if not answer.strip() or not question.strip():
        return 0.0
    vecs = embedder.embed([answer, question])
    return _cosine(vecs[0], vecs[1])


def context_support(answer: str, context_chunks: list[str], embedder: Embedder) -> float:
    """Max cosine similarity between the answer and any single retrieved
    chunk — NOT the mean. Faithfulness should ask "is there a chunk this
    answer could plausibly be grounded in", not be diluted by 9 irrelevant
    chunks sharing top-k with the 1 relevant one (the exact failure mode of
    the old Jaccard-over-the-whole-context metric).
    """
    if not answer.strip() or not context_chunks:
        return 0.0
    texts = [answer, *context_chunks]
    vecs: list[Any] = embedder.embed(texts)
    answer_vec, chunk_vecs = vecs[0], vecs[1:]
    return max((_cosine(answer_vec, cv) for cv in chunk_vecs), default=0.0)


def grounded_in_correct_source(
    answer: str,
    source_refs: list[dict[str, Any]],
    article_refs: list[str],
    embedder: Embedder,
) -> float | None:
    """Cosine similarity between the answer and SPECIFICALLY the chunk(s)
    matching the ground-truth article_refs — not the max over every
    retrieved chunk like context_support. Distinguishes "the model used the
    right material" from "the model used something semantically similar but
    wrong" or answered from parametric knowledge rather than the provided
    context — a real, measured pattern (Phase 1 funnel analysis, run
    44329138: 12/75 answerable questions had retrieval_recall_at_k=0.0 but
    answer_similarity/context_support 0.6-0.9, i.e. a confident-sounding
    answer with no actual grounding in retrieved material).

    Returns None — not 0.0 — when no retrieved chunk matches article_refs at
    all (recall_at_k=0): there is nothing to compute a "is the answer
    grounded in the correct chunk" score against, so the metric is
    undefined, not failing. Callers should omit the key rather than record
    a misleading zero (same convention as the answerability gating in
    core/eval/answerability.py).

    Found live: matched chunks used to be found via a hardcoded
    "{source_code}/{article_no}" lookup — the shape article_refs use for a
    corpus with a one-numbered-file-per-unit layout, but not what a corpus
    with no external document-code numbering scheme at all produces (see
    core/eval/retrieval_metrics.py#extract_ref_id's fallback tiers). Every
    chunk from a corpus like that failed the lookup regardless of whether
    retrieval actually found the right one, so this metric silently
    vanished (None) from every question of every such run. Now built via
    extract_ref_id, the same realm-agnostic ref-id construction
    retrieval_recall_at_k/precision_at_k already use — must stay in sync
    with it.
    """
    if not answer.strip():
        return None
    from core.eval.retrieval_metrics import extract_ref_id

    expected = set(article_refs)
    matching_chunks = [
        sr["chunk_text"]
        for sr in source_refs
        if sr.get("chunk_text") and extract_ref_id(sr) in expected
    ]
    if not matching_chunks:
        return None
    return context_support(answer, matching_chunks, embedder)
