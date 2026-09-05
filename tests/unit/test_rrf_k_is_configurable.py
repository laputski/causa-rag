"""The rank-fusion constant becomes a setting, and stays the one that was set.

Reciprocal rank fusion scores a document as 1/(rrf_k + rank). The constant
decides how steeply rank 1 outweighs rank 10, so it is the single number that
governs how much the top of each list dominates the merge. Sixty comes from the
paper that introduced the method and had been carried here unmeasured.

It had no field on a configuration at all, so no run recorded which value it
used and no comparison could say what changing it costs. That absence is the
recorded reason one atlas entry is undetectable, and this is what removes it.
"""
from __future__ import annotations

from typing import Any

from core.experiment.config import ComponentRef, ExperimentConfig
from core.experiment.runner import _rebind_merge
from core.retrieval.hybrid import HybridRetriever


class _Stub:
    def retrieve(self, **kwargs: Any) -> list:
        return []

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1]] * len(texts)


def _hybrid(**kwargs: Any) -> HybridRetriever:
    return HybridRetriever(dense_retriever=_Stub(), sparse_retriever=_Stub(),
                           embedder=_Stub(), **kwargs)


def _config(**fields: Any) -> ExperimentConfig:
    return ExperimentConfig(
        name="c",
        chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
        embedder=ComponentRef(kind="embedder", component_id="bge"),
        generator=ComponentRef(kind="generator", component_id="ollama"),
        **fields,
    )


# ── the field ─────────────────────────────────────────────────────────────────

def test_the_constant_is_a_field_of_a_configuration() -> None:
    assert _config(rrf_k=10).rrf_k == 10


def test_leaving_it_unset_leaves_every_historical_fingerprint_alone() -> None:
    """Runs made before the field existed have to keep comparing against runs
    made after it. The fingerprint excludes the field while it holds its
    default, which is the mechanism `_BACKCOMPAT_DEFAULTS` exists for; without
    the entry, every stored run would have become incomparable on the day the
    field landed."""
    from core.experiment.config import _BACKCOMPAT_DEFAULTS

    assert _BACKCOMPAT_DEFAULTS.get("rrf_k", "missing") is None
    unset = _config()
    payload = unset.model_dump(exclude={"config_hash"})
    payload.pop("rrf_k")
    import hashlib
    import json
    for key, default in _BACKCOMPAT_DEFAULTS.items():
        if payload.get(key) == default:
            payload.pop(key, None)
    without = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]
    assert unset.config_hash == without, "adding the field changed the fingerprint of an unset config"


def test_setting_it_changes_the_fingerprint() -> None:
    """The other half. A knob excluded from the fingerprint whatever its value
    would let two genuinely different runs pass for repeats of each other."""
    assert _config(rrf_k=10).config_hash != _config(rrf_k=20).config_hash
    assert _config(rrf_k=10).config_hash != _config().config_hash


# ── the application ───────────────────────────────────────────────────────────

def test_the_value_asked_for_is_the_value_the_retriever_gets() -> None:
    assert _rebind_merge(_hybrid(merge="rrf"), None, None, 200)._rrf_k == 200


def test_changing_the_merge_weight_no_longer_resets_the_constant() -> None:
    """Measured before the fix: a retriever built with rrf_k=10 came back from
    a merge-weight change carrying 60. Two settings on the same wrapper, and
    moving one silently returned the other to its default."""
    built = _hybrid(merge="rrf", alpha=0.5, rrf_k=10)
    assert _rebind_merge(built, "weighted", 0.7)._rrf_k == 10


def test_asking_for_nothing_keeps_what_was_built() -> None:
    assert _rebind_merge(_hybrid(merge="rrf", rrf_k=10), None, None, None)._rrf_k == 10


def test_a_retriever_already_carrying_the_value_is_not_rebuilt() -> None:
    """Rebuilding is not free and, more to the point, `_rebind_corpus_id` has
    already bound these retrievers to the right corpus by the time this runs;
    an unnecessary rebuild is an unnecessary chance to undo that."""
    built = _hybrid(merge="rrf", alpha=0.5, rrf_k=10)
    assert _rebind_merge(built, "rrf", 0.5, 10) is built


def test_a_dense_only_retriever_is_left_alone() -> None:
    """Nothing to merge, so nothing to say. Raising at a caller that merely
    passed its config along would fail runs over a field that cannot apply."""
    dense = _Stub()
    assert _rebind_merge(dense, "rrf", 0.5, 10) is dense


# ── that the setting has a consequence ────────────────────────────────────────

def _ranked(ids: list[str]) -> list:
    from core.models import Chunk, ScoredChunk
    return [ScoredChunk(chunk=Chunk(chunk_id=i, doc_id="d", text=i, structural_path="p"),
                        score=1.0, retriever_id="r") for i in ids]


class _Ranked(_Stub):
    def __init__(self, ids: list[str]) -> None:
        self._out = _ranked(ids)

    def retrieve(self, **kwargs: Any) -> list:
        return self._out


def test_the_constant_decides_which_document_wins() -> None:
    """A setting nothing observes is a setting nobody can stage a defect with,
    so the consequence is asserted and not assumed.

    X takes first place in one half and is absent from the other. Y sits sixth
    and seventh, meaning both halves agree on it weakly. At sixty the two lists
    have nearly equal say and the agreement wins; at one the single first place
    outweighs everything below it, which is the merge ceasing to be a merge.

    A symmetric fixture was tried first and moved nothing: both constants
    returned the same order, and only the score spread differed. It would have
    passed as evidence for a setting that changed nothing observable.
    """
    dense = _Ranked(["X", "a", "b", "c", "d", "Y", "e"])
    sparse = _Ranked(["p", "q", "r", "s", "t", "u", "Y"])

    def top_two(rrf_k: int) -> list[str]:
        merged = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse,
                                 embedder=_Stub(), merge="rrf", rrf_k=rrf_k).retrieve("q", k=2)
        return [sc.chunk.chunk_id for sc in merged]

    assert top_two(60) == ["Y", "X"], "the platform default no longer lets agreement outweigh one first place"
    assert top_two(1) == ["X", "p"], "pinning the constant to one no longer changes the order"
