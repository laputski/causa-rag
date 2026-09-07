"""Resolving a golden question's `article_refs` against what is actually
INDEXED, rather than against files on some disk path.

Why this module exists. A control question declares which source unit
should answer it (`article_refs`). Deciding whether the corpus can answer
it at all is a property of (refs, corpus) and gates every retrieval metric
downstream: an "uncovered" question is excluded from recall entirely. That
decision used to be made by looking for files under a hardcoded
a corpus path baked into `services/api_gateway/routers/
experiments.py` — one Realm's own corpus, checked for every Realm's run,
against the platform's own filesystem rather than the index the served
system actually searches. the design notes records four separate incidents
caused by that coupling; this module removes its cause rather than patching
a fifth symptom.

Three-valued, deliberately. `presence()` returns `unknown` as a distinct
outcome from `absent`, because "I could not check" and "I checked, it is
not there" have opposite consequences and were previously indistinguishable
— a missing/unreachable corpus silently classified every question
`uncovered`, which zeroes out retrieval metrics with no error anywhere. A
boolean cannot express that difference, so it is not a boolean.

Ref ids are built by `core/eval/retrieval_metrics.py#extract_ref_id`, the
same realm-agnostic construction `retrieval_recall_at_k`/`precision_at_k`/
`grounded_in_correct_source`/`citation_number_coverage` already use. Using
it here is what makes "is this ref covered" consistent-by-construction with
"did retrieval find this ref" — two questions that must never disagree
about what counts as the same source unit.

Pure logic: no I/O, no storage, no HTTP. The caller supplies already-read
chunks (`services/api_gateway/routers/experiments.py` reads them out of the
index); this keeps `core/` free of the adapter dependencies the layer
isolation fitness function forbids (the design notes).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from core.eval.retrieval_metrics import extract_ref_id

Presence = Literal["present", "absent", "unknown"]


@runtime_checkable
class RefResolver(Protocol):
    """Answers whether one `article_refs` entry corresponds to something the
    served system can actually retrieve."""

    def presence(self, ref: str) -> Presence: ...

    @property
    def checked(self) -> bool:
        """False when this resolver cannot verify anything at all, so a
        caller can report "coverage was not checked" instead of presenting
        an unverified guess as a verified result."""
        ...


@dataclass(frozen=True)
class IndexRefResolver:
    """Resolves against a prebuilt set of ref ids read out of the index.

    Membership, not similarity: a ref matches only when the index contains a
    chunk whose `extract_ref_id` is byte-identical. For `article_no`-shaped
    refs that is stable (a published article number rarely changes). For
    `structural_path`-shaped refs it inherits the same staleness
    `extract_ref_id`'s own docstring flags — re-ingesting a document after
    its headings were edited shifts the path and therefore the ref id. That
    is a property of the ref scheme, not of this resolver, and it is shared
    with `retrieval_recall_at_k`, so both move together.
    """

    known_ref_ids: frozenset[str]
    # How many chunks each source unit occupies in the index.
    # Optional so an older caller (or a test) can build a resolver from ids
    # alone; `chunk_count` then returns None, which the root-cause split
    # reads as "cannot tell" rather than as "one chunk".
    ref_chunk_counts: Mapping[str, int] | None = None

    def presence(self, ref: str) -> Presence:
        return "present" if ref in self.known_ref_ids else "absent"

    def chunk_count(self, ref: str) -> int | None:
        """How many chunks this source unit was split into, or None when the
        resolver was built without counts.

        None and 0 are different answers and must not be confused: None means
        nobody counted, 0 would mean the unit is not in the index at all,
        which `presence` already says.
        """
        if self.ref_chunk_counts is None:
            return None
        return self.ref_chunk_counts.get(ref, 0)

    @property
    def checked(self) -> bool:
        return True


@dataclass(frozen=True)
class UnknownRefResolver:
    """Used whenever coverage genuinely cannot be established: an external
    RAG whose corpus the platform never sees, an unreachable index, a run
    configured without one.

    Returning `unknown` (never `absent`) is the whole point — it makes a
    failure to check degrade into "not verified" instead of into a false
    "not covered" that would silently drop the question out of every
    retrieval metric.
    """

    #: Which of the ways coverage could not be checked this is, in one word
    #: the interface can look up. The sentence beside it reaches a reader
    #: inside a finding, and a finding is read in the reader's own language.
    #: There are five of these and each is a decision this platform made, so
    #: an identifier costs nothing and is what makes the sentence
    #: translatable.
    reason_id: str = "no_resolver_configured"
    reason: str = "no resolver configured"
    #: What a library said, when a library is why. Never ours to translate and
    #: never composed here: an exception's own message, shown as it arrived.
    note: str = ""

    def presence(self, ref: str) -> Presence:
        return "unknown"

    @property
    def checked(self) -> bool:
        return False


def ref_source_from_chunk(chunk: Any) -> dict[str, Any]:
    """Flattens a `core/models.py#Chunk` into the shape `extract_ref_id`
    expects.

    Needed because `source_code`/`article_no` live inside `chunk.metadata`,
    while `doc_id`/`structural_path` are top-level fields —
    `core/pipeline.py#_to_source_refs` performs the identical lift when
    building `SourceRef`s, and this mirrors it so a chunk read from the
    index yields the same ref id it would yield after passing through the
    pipeline. Accepts plain dicts too, so tests need no model objects.
    """
    if isinstance(chunk, dict):
        metadata = chunk.get("metadata") or {}
        return {
            "source_code": chunk.get("source_code") or metadata.get("source_code"),
            "article_no": chunk.get("article_no") or metadata.get("article_no"),
            "structural_path": chunk.get("structural_path"),
            "doc_id": chunk.get("doc_id"),
        }
    metadata = getattr(chunk, "metadata", None) or {}
    return {
        "source_code": metadata.get("source_code"),
        "article_no": metadata.get("article_no"),
        "structural_path": getattr(chunk, "structural_path", None),
        "doc_id": getattr(chunk, "doc_id", None),
    }


def count_ref_chunks(chunks: Any) -> dict[str, int]:
    """How many chunks each source unit occupies.

    A unit split across several chunks is a candidate explanation
    for "the expected source is indexed but retrieval does not surface it":
    each fragment carries only part of the unit, so none of them may be close
    to a question the whole unit answers. Counted in the same pass the ref
    index is built from, so it costs nothing extra.
    """
    counts: dict[str, int] = {}
    for chunk in chunks:
        ref_id = extract_ref_id(ref_source_from_chunk(chunk))
        if ref_id:
            counts[ref_id] = counts.get(ref_id, 0) + 1
    return counts


def build_ref_index(chunks: Any) -> frozenset[str]:
    """Ref ids of every chunk that has one. Chunks whose metadata yields no
    ref id at all (`extract_ref_id` returns None) are skipped rather than
    stored as empty strings, so an unidentifiable chunk never makes an
    unrelated ref look present."""
    return frozenset(count_ref_chunks(chunks))


def resolver_from_chunks(chunks: Any) -> IndexRefResolver:
    counts = count_ref_chunks(chunks)
    return IndexRefResolver(known_ref_ids=frozenset(counts), ref_chunk_counts=counts)
