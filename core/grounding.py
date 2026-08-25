"""Grounding check — claim-level verification against retrieved context."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from core.models import GroundingResult, ScoredChunk


@dataclass
class ClaimCheck:
    claim: str
    supported: bool
    evidence: str = ""


@runtime_checkable
class ClaimGrounder(Protocol):
    grounder_id: str

    def check(
        self,
        answer_text: str,
        context_chunks: list[ScoredChunk],
    ) -> GroundingResult:
        ...


def _split_claims(text: str) -> list[str]:
    """Split text into sentence-level claims."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


class TokenOverlapGrounder:
    """Stub grounder — uses token overlap between claim and context to assess support.

    A claim is considered supported if ≥ ``threshold`` fraction of its tokens
    appear in the combined context text.
    """

    grounder_id = "token_overlap"

    def __init__(self, threshold: float = 0.3) -> None:
        self._threshold = threshold

    def check(
        self,
        answer_text: str,
        context_chunks: list[ScoredChunk],
    ) -> GroundingResult:
        if not context_chunks:
            return GroundingResult(
                is_grounded=False,
                unsupported_claims=_split_claims(answer_text),
                confidence=0.0,
            )

        context_tokens = set(
            " ".join(sc.chunk.text for sc in context_chunks).lower().split()
        )

        claims = _split_claims(answer_text)
        unsupported: list[str] = []

        for claim in claims:
            tokens = set(claim.lower().split())
            if not tokens:
                continue
            overlap = len(tokens & context_tokens) / len(tokens)
            if overlap < self._threshold:
                unsupported.append(claim)

        supported_count = len(claims) - len(unsupported)
        confidence = supported_count / len(claims) if claims else 1.0

        return GroundingResult(
            is_grounded=len(unsupported) == 0,
            unsupported_claims=unsupported,
            confidence=confidence,
        )
