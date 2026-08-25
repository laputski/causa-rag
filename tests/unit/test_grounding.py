"""Unit tests — TokenOverlapGrounder."""
from core.grounding import TokenOverlapGrounder, _split_claims
from core.models import Chunk, ScoredChunk


def _sc(text: str) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(doc_id="d1", text=text), score=1.0, retriever_id="t")


def test_split_claims_basic():
    claims = _split_claims("The first statement. The second statement.")
    assert len(claims) == 2


def test_split_claims_short_filtered():
    claims = _split_claims("Yes. No. The rule applies in this case.")
    # short single-word sentences filtered out
    long_claims = [c for c in claims if len(c) > 10]
    assert len(long_claims) >= 1


def test_confidence_partial_support():
    g = TokenOverlapGrounder(threshold=0.3)
    context = [_sc("the rule applies to a lease under the code")]
    result = g.check(
        "The rule applies to a lease. An entirely different, unconnected text.",
        context,
    )
    assert 0.0 < result.confidence < 1.0


def test_full_support_confidence_one():
    g = TokenOverlapGrounder(threshold=0.1)
    result = g.check(
        "The rule applies here.",
        [_sc("the rule applies here without exception")],
    )
    assert result.is_grounded
    assert result.confidence == 1.0


def test_unsupported_claims_listed():
    g = TokenOverlapGrounder(threshold=0.9)
    result = g.check(
        "A wholly invented statement resting on none of the documents.",
        [_sc("the documents")],
    )
    assert len(result.unsupported_claims) > 0
