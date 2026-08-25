"""Per-question funnel diagnosis (Eval Measurement Trustworthiness, Phase
1) — answers "on which layer (corpus → retrieval → rerank → generation)
did THIS question's result go wrong", synthesizing signals that already
exist (answerability class, retrieval_recall_at_k, answer_similarity)
plus two new ones (pre-rerank recall, grounded_in_correct_source) into one
verdict, instead of requiring the engineer to manually correlate four
fields per question.

Thresholds reused verbatim from ui/src/pages/RunPage.tsx's per-metric
good/warn bands — same language for the aggregate metric cards and the
per-question funnel verdict, not a second ad-hoc scale.

Deliberately does NOT use retrieval_precision_at_k in the primary verdict —
it's a secondary, already-visible signal (noisy top-k), not a layer
attribution input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Layer = Literal[
    "not_applicable",
    "suspected_ungrounded_answer",
    "retrieval",
    "rerank",
    "generation",
    "ok",
]

# Mirrors ui/src/pages/RunPage.tsx meta.good/meta.warn for
# retrieval_recall_at_k / answer_similarity.
_RECALL_GOOD = 0.6
_RECALL_WARN = 0.3
_SIMILARITY_WARN = 0.4
_SIMILARITY_SUSPICIOUS = 0.6  # "confidently good-sounding" threshold for the combined check
_GROUNDED_WARN = 0.4


@dataclass
class FunnelVerdict:
    layer: Layer
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"layer": self.layer, "detail": self.detail}


def diagnose_question(
    answerability: str,
    metrics: dict[str, float],
    pre_rerank_recall_at_k: float | None = None,
) -> FunnelVerdict:
    """Pure function — no I/O, no LLM. `metrics` is a question's
    services/api_gateway/routers/experiments.py:_CompositeEvaluator output
    dict (retrieval_recall_at_k, answer_similarity, grounded_in_correct_source,
    ...). `pre_rerank_recall_at_k` is computed by the caller from
    Answer.pre_rerank_source_refs (core/pipeline.py) when a reranker ran;
    None otherwise (nothing to distinguish — see core/pipeline.py docstring
    on pre_rerank_source_refs).
    """
    if answerability != "answerable":
        reason = (
            "the question falls outside the corpus domain (out_of_scope)" if answerability == "out_of_scope"
            else "the reference source is absent from the corpus (uncovered)"
        )
        return FunnelVerdict(
            layer="not_applicable",
            detail=f"Outside the diagnosable funnel: {reason}. Only correct_refusal is scored.",
        )

    recall = metrics.get("retrieval_recall_at_k")
    similarity = metrics.get("answer_similarity")
    grounded = metrics.get("grounded_in_correct_source")

    # Checked FIRST, before any per-layer verdict: low/no recall but a
    # confidently good-sounding answer is the empirically-confirmed failure
    # mode (run 44329138: 12/75 answerable questions, recall_at_k=0 with
    # answer_similarity 0.6-0.9) that isolated per-layer checks would
    # mis-read as "retrieval bad, generation fine".
    if recall is not None and recall < _RECALL_WARN and similarity is not None and similarity >= _SIMILARITY_SUSPICIOUS:
        return FunnelVerdict(
            layer="suspected_ungrounded_answer",
            detail=(
                f"retrieval_recall_at_k={recall:.2f} is low while answer_similarity={similarity:.2f} "
                "is high. The answer resembles the right one yet probably does not rest on the "
                "retrieved context, drawing instead on the model's parametric knowledge."
            ),
        )

    if recall is not None and recall < _RECALL_WARN:
        if (
            pre_rerank_recall_at_k is not None
            and pre_rerank_recall_at_k >= _RECALL_GOOD
            and pre_rerank_recall_at_k > recall
        ):
            return FunnelVerdict(
                layer="rerank",
                detail=(
                    f"recall@k was {pre_rerank_recall_at_k:.2f} before reranking and {recall:.2f} "
                    "after: retrieval found the required source and the reranker dropped it from top-k."
                ),
            )
        return FunnelVerdict(
            layer="retrieval",
            detail=f"retrieval_recall_at_k={recall:.2f}: the required source never reached the final context.",
        )

    if grounded is not None and grounded < _GROUNDED_WARN:
        return FunnelVerdict(
            layer="generation",
            detail=(
                f"the required source was in the context (recall@k={recall:.2f}, with a grounded signal "
                f"present) yet the answer rests on it weakly (grounded_in_correct_source={grounded:.2f})."
            ),
        )

    if similarity is not None and similarity < _SIMILARITY_WARN:
        return FunnelVerdict(
            layer="generation",
            detail=f"answer_similarity={similarity:.2f} is low even though retrieval found the required source.",
        )

    return FunnelVerdict(layer="ok", detail="Every funnel layer looks healthy for this question.")
