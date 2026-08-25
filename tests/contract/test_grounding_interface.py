"""Contract tests — ClaimGrounder Protocol compliance."""
from core.grounding import ClaimGrounder, TokenOverlapGrounder
from core.models import Chunk, ScoredChunk


def _sc(text: str) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(doc_id="d1", text=text), score=1.0, retriever_id="t")


def test_stub_is_claim_grounder():
    assert isinstance(TokenOverlapGrounder(), ClaimGrounder)


def test_no_context_returns_not_grounded():
    g = TokenOverlapGrounder()
    result = g.check("This rule applies.", [])
    assert not result.is_grounded
    assert result.confidence == 0.0


def test_matching_context_grounds():
    g = TokenOverlapGrounder(threshold=0.2)
    result = g.check(
        "This rule applies to lease agreements.",
        [_sc("the rule applies to lease agreements under the code")],
    )
    assert result.is_grounded


def test_unrelated_context_not_grounded():
    g = TokenOverlapGrounder(threshold=0.8)
    result = g.check(
        "A special rule of the code applies.",
        [_sc("an entirely different subject with no overlap")],
    )
    assert not result.is_grounded
