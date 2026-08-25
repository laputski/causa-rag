"""A standalone, third-party-style RAG implementation.

Deliberately has ZERO imports from this platform — it represents the kind
of system someone clones into their own repo and wants to debug with this
platform's diagnostics, without rewriting it. See adapter.py for the <=20
lines that wire it into core.sdk.
"""
from __future__ import annotations

_CORPUS = [
    "Cats sleep for 12 to 16 hours a day on average.",
    "Dogs were domesticated roughly 15,000 years ago.",
    "Cats use their whiskers to judge the width of a gap.",
    "Parrots can imitate human speech.",
]


def search(query: str, top_n: int) -> list[tuple[str, float]]:
    """The external system's own retrieval function — naive keyword overlap."""
    query_words = set(query.lower().split())
    scored = []
    for doc in _CORPUS:
        doc_words = set(doc.lower().split())
        overlap = len(query_words & doc_words)
        if overlap > 0:
            scored.append((doc, float(overlap)))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_n]


def answer(prompt: str) -> str:
    """The external system's own generation function — rule-based, no LLM call."""
    # The prompt is assembled by whatever pipeline calls this, and the marker
    # before the question depends on which prompt template that pipeline used.
    # Both known forms are accepted; an unknown one falls through to the whole
    # prompt rather than to an empty answer.
    for marker in ("Question:", "Вопрос:"):
        if marker in prompt:
            return "Based on the retrieved documents: " + prompt.split(marker)[-1].strip()
    return "Based on the retrieved documents: " + prompt.strip()
