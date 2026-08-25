"""Corpus health — preventive diagnostics on the indexed corpus.

Mirrors core/eval/detectors.py (run-level silent degradation) but operates on
the raw indexed chunks themselves, before any experiment is run — catching
"chunks are header-only" or "duplicate chunks" while inspecting the index, not
after a bad run.

``analyze()`` is pure / deterministic — no LLM, no I/O (callers fetch chunks
first). ``detect_near_duplicates()`` is the one exception — it needs a live
Qdrant connection (k-NN search against vectors already computed at ingest
time, see its docstring for why this doesn't need a new vector index), so it
is a separate entry point the caller invokes explicitly, not part of the
otherwise-pure ``analyze()``.
"""
from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

Severity = str  # "ok" | "info" | "warn" | "error"

# Matches a leading numeric identifier in a structural leaf segment, e.g.
# "article[Article 47. Foo]" -> "47", "section[12-1]" -> "12-1". Domain-neutral:
# no heading word is hardcoded, so any numbered structural unit works.
_LEAF_NUMBER_RE = re.compile(r"(\d+(?:-\d+)?)")


@dataclass
class HealthItem:
    id: str
    severity: Severity
    title: str
    detail: str
    action: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "action": self.action,
        }


@dataclass
class CorpusHealth:
    n_chunks: int
    n_duplicates: int
    n_header_only: int
    avg_length: float
    min_length: int
    max_length: int
    items: list[HealthItem]
    n_duplicate_numbers: int = 0
    n_missing_numbers: int = 0
    language_distribution: dict[str, int] = field(default_factory=dict)
    language_sample_size: int = 0
    # Ten buckets over [min_length, max_length]. A corpus is judged by the
    # shape of its length distribution, not by its mean: a mean of 700 is the
    # same number whether every chunk is 700 characters or half are headings
    # of 80 and half are walls of 1300, and only the second is a problem.
    length_deciles: list[int] = field(default_factory=list)
    # Chunks whose structural path was never parsed. `root` is what the
    # chunker writes when a document has no tree, so it counts as absent —
    # treating it as a path reports zero on a corpus that is entirely
    # unstructured.
    n_missing_path: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_chunks": self.n_chunks,
            "n_duplicates": self.n_duplicates,
            "n_header_only": self.n_header_only,
            "avg_length": round(self.avg_length, 1),
            "min_length": self.min_length,
            "max_length": self.max_length,
            "n_duplicate_numbers": self.n_duplicate_numbers,
            "n_missing_numbers": self.n_missing_numbers,
            "length_deciles": self.length_deciles,
            "n_missing_path": self.n_missing_path,
            "language_distribution": self.language_distribution,
            "language_sample_size": self.language_sample_size,
            "items": [i.to_dict() for i in self.items],
        }


_HEADER_ONLY_LEN = 40
# langdetect itself is fast (pure Python, no model download) but calling it
# on every chunk of a 35k-chunk corpus adds up — sampled, like every other
# "deep" signal in this module, and explicitly labeled as a sample in the
# resulting HealthItem rather than presented as an exhaustive count.
_LANGUAGE_SAMPLE_SIZE = 300
_LANGUAGE_MIN_TEXT_LEN = 20  # langdetect is unreliable on very short strings


def _deciles(lengths: list[int], buckets: int = 10) -> list[int]:
    """Counts per equal-width bucket over [min, max].

    Equal width, not equal count: the question the histogram answers is "are
    there two clumps" — headings against bodies — and equal-count buckets
    would flatten exactly that into ten identical bars.
    """
    if not lengths:
        return []
    lo, hi = min(lengths), max(lengths)
    if hi == lo:
        return [len(lengths)] + [0] * (buckets - 1)
    out = [0] * buckets
    for n in lengths:
        i = min(buckets - 1, int((n - lo) / (hi - lo) * buckets))
        out[i] += 1
    return out


def analyze(chunks: list[Any]) -> CorpusHealth:
    """Compute counters + detector items over a (sampled) list of indexed chunks.

    Each chunk is expected to expose ``.text`` and ``.structural_path`` (the
    ``core.models.Chunk`` shape) but duck-typing is fine for dict-like inputs too.
    """
    def _text(c: Any) -> str:
        return (c.get("text") if isinstance(c, dict) else getattr(c, "text", "")) or ""

    def _path(c: Any) -> str:
        if isinstance(c, dict):
            return c.get("structural_path") or ""
        return getattr(c, "structural_path", "") or ""

    texts = [_text(c) for c in chunks]
    n_chunks = len(chunks)
    n_duplicates = n_chunks - len(set(texts)) if texts else 0
    lengths = [len(t) for t in texts] if texts else [0]
    avg_length = sum(lengths) / len(lengths)
    header_only = [c for c in chunks if _path(c) and len(_text(c).strip()) < _HEADER_ONLY_LEN]
    n_header_only = len(header_only)
    n_missing_path = sum(1 for c in chunks if _path(c) in ("", "root"))
    length_deciles = _deciles(lengths)

    # A numbered structural unit (leaf segment of structural_path) that maps to
    # more than one distinct leaf string signals a source-corpus numbering
    # collision — e.g. two different articles both labelled "47" because the
    # per-file export silently dropped/overwrote one of them. Domain-neutral:
    # operates on whatever numeric id appears in the leaf, no domain terms.
    leaf_numbers: dict[str, set[str]] = {}
    for c in chunks:
        path = _path(c)
        if not path:
            continue
        leaf = path.rsplit("/", 1)[-1]
        m = _LEAF_NUMBER_RE.search(leaf)
        if not m:
            continue
        leaf_numbers.setdefault(m.group(1), set()).add(leaf)
    dup_numbers = {num: leaves for num, leaves in leaf_numbers.items() if len(leaves) > 1}
    n_duplicate_numbers = len(dup_numbers)

    # Completeness check: a gap in an otherwise sequential numbering scheme
    # signals a structural unit that exists in the source but never made it
    # into the index — e.g. a malformed heading the parser couldn't capture,
    # or a file silently dropped during corpus export. Only meaningful when
    # the corpus is clearly numbered (>=2 distinct numbers); skipped (rather
    # than flagged) when most of the range is absent, since that just means
    # the corpus isn't sequentially numbered at all (no signal either way).
    covered_ints = {int(n.split("-")[0]) for n in leaf_numbers}
    missing_numbers: list[int] = []
    if len(covered_ints) >= 2:
        lo, hi = min(covered_ints), max(covered_ints)
        span = hi - lo + 1
        gaps = sorted(set(range(lo, hi + 1)) - covered_ints)
        if gaps and len(gaps) / span <= 0.5:
            missing_numbers = gaps
    n_missing_numbers = len(missing_numbers)

    # Language mix — on a sample (see _LANGUAGE_SAMPLE_SIZE). A corpus that's
    # supposed to be one language but isn't usually means either genuinely
    # mixed-language source documents (fine, just worth knowing) or chunking
    # garbage producing text langdetect can't classify confidently (less
    # fine — see "und"/error handling below).
    language_distribution: dict[str, int] = {}
    eligible_for_lang = [t for t in texts if len(t.strip()) >= _LANGUAGE_MIN_TEXT_LEN]
    lang_sample = (
        random.sample(eligible_for_lang, _LANGUAGE_SAMPLE_SIZE)
        if len(eligible_for_lang) > _LANGUAGE_SAMPLE_SIZE
        else eligible_for_lang
    )
    if lang_sample:
        from langdetect import LangDetectException, detect

        counts: Counter[str] = Counter()
        for t in lang_sample:
            try:
                counts[detect(t)] += 1
            except LangDetectException:
                counts["unknown"] += 1
        language_distribution = dict(counts)
    language_sample_size = len(lang_sample)

    items: list[HealthItem] = []

    if n_chunks == 0:
        items.append(HealthItem(
            id="empty_corpus", severity="error", title="The corpus is empty",
            detail="The index holds no chunks at all.",
            action="Load a corpus through Data → Corpus.",
        ))
        return CorpusHealth(0, 0, 0, 0.0, 0, 0, items)

    if n_duplicates > 0:
        items.append(HealthItem(
            id="duplicates", severity="warn", title="Duplicates in the index",
            detail=f"{n_duplicates}/{n_chunks} chunks carry identical text.",
            action="Check that chunk_id is deterministic, then re-index.",
        ))

    if n_chunks and n_header_only / n_chunks > 0.3:
        items.append(HealthItem(
            id="header_only", severity="warn", title="Many heading-only chunks",
            detail=f"{n_header_only}/{n_chunks} chunks carry almost no text (structural_path only).",
            action="Raise min_chars, or switch to sentence or paragraph chunking.",
        ))

    if n_chunks and n_missing_path == n_chunks:
        items.append(HealthItem(
            id="no_structure", severity="warn",
            title="No structural path was parsed for any chunk",
            detail=f"All {n_chunks} chunks carry no structural path.",
            action="Enable a domain parser, or expect source references to be unavailable.",
        ))
    elif n_chunks and n_missing_path / n_chunks > 0.2:
        items.append(HealthItem(
            id="some_missing_path", severity="warn",
            title="Chunks without a structural path",
            detail=f"{n_missing_path}/{n_chunks} chunks carry no structural path.",
            action="A source reference cannot be built for them; check the parser.",
        ))

    if avg_length < 50:
        items.append(HealthItem(
            id="too_short", severity="warn", title="Chunks are very short",
            detail=f"The average chunk is {avg_length:.0f} characters, which leaves generation little context.",
            action="Raise chunk_size, or review the chunking strategy.",
        ))

    if dup_numbers:
        sample = ", ".join(sorted(dup_numbers, key=lambda n: int(n.split("-")[0]))[:10])
        items.append(HealthItem(
            id="duplicate_structural_numbers", severity="warn",
            title="Structural numbers repeat",
            detail=(
                f"{n_duplicate_numbers} numbers appear on nodes with different content "
                f"(for example: {sample}). That is expected when one corpus_id holds several "
                f"documents, each numbered from its own start; splitting them into separate "
                f"corpus_ids resolves it. With a single document it can mean that the export "
                f"gave one number to two different nodes, so check by hand whether one "
                f"version's content is being lost."
            ),
            action="For a single document, compare the aggregated source against the "
                   "per-file export at these numbers. For several documents, split them "
                   "into separate corpus_ids.",
        ))

    if missing_numbers:
        sample = ", ".join(str(n) for n in missing_numbers[:10])
        items.append(HealthItem(
            id="missing_structural_numbers", severity="warn",
            title="Gaps in the structural numbering",
            detail=(
                f"{n_missing_numbers} numbers are absent inside the range "
                f"{min(covered_ints)}–{max(covered_ints)} (for example: {sample}). "
                f"Either the node is genuinely absent from the source, which makes the gap "
                f"expected, or it exists there and never reached the index because the parser "
                f"missed its heading or the file was lost during export."
            ),
            action="Compare the listed numbers against the corpus source files. Where a file "
                   "exists but is not indexed, check its heading format and re-index.",
        ))

    # Flag only when a second language has a non-trivial share — a single
    # stray misclassification (langdetect is never 100% accurate on short
    # legal text) shouldn't read as "mixed corpus".
    if language_distribution and language_sample_size:
        sorted_langs = sorted(language_distribution.items(), key=lambda kv: -kv[1])
        if len(sorted_langs) > 1 and sorted_langs[1][1] / language_sample_size > 0.05:
            dist_str = ", ".join(f"{lang}: {n}" for lang, n in sorted_langs[:5])
            items.append(HealthItem(
                id="mixed_language", severity="info",
                title="The corpus spans several languages",
                detail=(
                    f"Across a sample of {language_sample_size} chunks: {dist_str}. "
                    f"That is expected when a corpus holds documents in more than one "
                    f"language. It does change how the graph's RELATED keywords link chunks: "
                    f"documents in different languages share no vocabulary, so they cluster "
                    f"separately (visible on the Graph tab)."
                ),
            ))

    if not items:
        items.append(HealthItem(
            id="ok", severity="ok", title="The corpus looks healthy",
            detail=f"{n_chunks} chunks, no duplicates, average length {avg_length:.0f} characters.",
        ))

    return CorpusHealth(
        n_chunks=n_chunks,
        n_duplicates=n_duplicates,
        n_header_only=n_header_only,
        avg_length=avg_length,
        min_length=min(lengths),
        max_length=max(lengths),
        items=items,
        n_duplicate_numbers=n_duplicate_numbers,
        n_missing_numbers=n_missing_numbers,
        language_distribution=language_distribution,
        language_sample_size=language_sample_size,
        length_deciles=length_deciles,
        n_missing_path=n_missing_path,
    )


def detect_near_duplicates(
    qdrant: Any,
    sample_chunk_ids: list[str],
    threshold: float = 0.97,
    max_pairs_reported: int = 10,
) -> HealthItem | None:
    """Near-duplicate detection via the corpus's OWN already-computed
    embeddings — not a new vector index (FAISS/MinHash etc.). Chunks are
    already embedded and stored in Qdrant at ingest time, so a k-NN search
    using an existing point as the query (``query=point_id``, Qdrant looks
    up that point's stored vector itself) is nearly free and needs no new
    infrastructure.

    This catches what exact-text dedup (the ``duplicates`` HealthItem
    above) can't: legitimate near-duplicate content — e.g. boilerplate
    clauses legislators repeat verbatim across different articles with only
    a name/office substituted (confirmed live on this corpus: articles on
    election-result certification for President vs. Parliament vs. local
    council deputies share nearly identical wording). Exact-string dedup
    sees these as distinct (correctly — they ARE different articles); this
    check additionally surfaces the similarity for a human to judge whether
    it's expected legislative boilerplate or an actual indexing mistake.

    ``sample_chunk_ids`` should already be a sample (caller's
    responsibility, see ``_LANGUAGE_SAMPLE_SIZE`` for the same reasoning) —
    a full scan is one Qdrant round-trip per chunk, impractical at corpus
    scale for what's meant to be a fast preventive check.
    """
    if not sample_chunk_ids:
        return None
    pairs: list[tuple[str, str, float]] = []
    seen_pairs: set[frozenset[str]] = set()
    for chunk_id in sample_chunk_ids:
        try:
            hits = qdrant._client.query_points(
                collection_name=qdrant._collection, query=chunk_id, limit=2, with_payload=False,
            ).points
        except Exception:
            continue  # point may have been deleted/re-indexed since the sample was drawn
        for hit in hits:
            neighbor_id = str(hit.id)
            if neighbor_id == chunk_id or hit.score < threshold:
                continue
            key = frozenset((chunk_id, neighbor_id))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            pairs.append((chunk_id, neighbor_id, hit.score))

    if not pairs:
        return None
    sample = ", ".join(f"{a[:8]}…/{b[:8]}… ({s:.2f})" for a, b, s in pairs[:max_pairs_reported])
    return HealthItem(
        id="near_duplicates", severity="info",
        title="Near-duplicate chunks in the sample",
        detail=(
            f"{len(pairs)} chunk pair(s) out of a sample of {len(sample_chunk_ids)} score above "
            f"cosine {threshold} (for example: {sample}). This can be legitimate, as when a "
            f"source repeats one formulation across sections with only a name changed, or it "
            f"can mean content was duplicated by accident during ingest."
        ),
        action="Open the pairs by chunk_id through Data → Corpus → Content and judge each "
               "by hand: a legitimately repeated passage, or an indexing mistake.",
    )
