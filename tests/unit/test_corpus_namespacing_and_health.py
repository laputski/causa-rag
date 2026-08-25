"""corpus_id namespacing + corpus health."""
from __future__ import annotations

from adapters.opensearch import _index_name
from adapters.qdrant import _collection_name
from core.eval import corpus_health as ch
from core.experiment.config import ComponentRef, ExperimentConfig

# ── namespacing ────────────────────────────────────────────────────────────────

def test_default_corpus_keeps_original_collection_name():
    assert _collection_name("structure_aware", "bge_m3") == "structure_aware__bge_m3"
    assert _collection_name("structure_aware", "bge_m3", "default") == "structure_aware__bge_m3"


def test_non_default_corpus_gets_prefixed_collection():
    assert _collection_name("structure_aware", "bge_m3", "acme_v2") == "acme_v2__structure_aware__bge_m3"


def test_default_corpus_keeps_original_index_name():
    assert _index_name("fixed") == "rag__fixed"
    assert _index_name("fixed", "default") == "rag__fixed"


# ── realm_id namespacing (two Realms sharing one physical Qdrant/
# OpenSearch instance used to collide on identical collection/index names) ──

def test_no_realm_id_leaves_collection_name_unchanged():
    """Omitting realm_id entirely (not even ''/None passed explicitly by a
    caller that doesn't know about Realms yet) must be byte-for-byte the
    the older name, which is what keeps every existing physical collection
    resolvable without a migration for callers that never adopt realm_id."""
    assert _collection_name("structure_aware", "bge_m3", "acme_v2") == "acme_v2__structure_aware__bge_m3"
    assert _index_name("fixed", "acme_v2") == "rag__acme_v2__fixed"


def test_realm_id_prefixes_default_corpus_collection():
    assert _collection_name("structure_aware", "bge_m3", "default", "demo") == "demo__structure_aware__bge_m3"
    assert _index_name("fixed", "default", "demo") == "rag__demo__fixed"


def test_realm_id_prefixes_non_default_corpus_collection():
    assert (
        _collection_name("structure_aware", "bge_m3", "handbook", "demo")
        == "demo__handbook__structure_aware__bge_m3"
    )
    assert _index_name("fixed", "handbook", "demo") == "rag__demo__handbook__fixed"


def test_different_realms_get_different_collection_names_for_same_corpus_id():
    """The actual bug this closes: two Realms both using corpus_id
    'default' against one shared Qdrant used to resolve to the identical
    collection name — Content/Health/Graph for one Realm silently showed
    the other Realm's data."""
    realm_a = _collection_name("structure_aware", "bge_m3", "default", "demo")
    realm_b = _collection_name("structure_aware", "bge_m3", "default", "acme")
    assert realm_a != realm_b


def test_non_default_corpus_gets_prefixed_index():
    assert _index_name("fixed", "acme_v2") == "rag__acme_v2__fixed"


# ── config_hash backward compatibility ───────────────

def _base() -> dict:
    return dict(
        name="t",
        chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
        embedder=ComponentRef(kind="embedder", component_id="bge"),
        generator=ComponentRef(kind="generator", component_id="ollama"),
    )


def test_default_corpus_id_keeps_prior_hash():
    old = ExperimentConfig(**_base())
    explicit = ExperimentConfig(**_base(), corpus_id="default")
    assert old.config_hash == explicit.config_hash


def test_non_default_corpus_id_changes_hash():
    old = ExperimentConfig(**_base())
    other = ExperimentConfig(**_base(), corpus_id="acme_v2")
    assert old.config_hash != other.config_hash


# ── corpus health ────────────────────────────────────────────────────────────

class _C:
    def __init__(self, text, structural_path=""):
        self.text = text
        self.structural_path = structural_path


def test_empty_corpus_flagged():
    h = ch.analyze([])
    assert h.n_chunks == 0
    assert any(i.id == "empty_corpus" for i in h.items)


def test_healthy_corpus_no_warnings():
    chunks = [_C(f"substantive text of article number {i} " * 3, f"section[{i}]") for i in range(5)]
    h = ch.analyze(chunks)
    assert h.n_duplicates == 0
    assert all(i.severity != "warn" for i in h.items)
    assert any(i.id == "ok" for i in h.items)


def test_duplicates_detected():
    chunks = [_C("identical text") for _ in range(5)]
    h = ch.analyze(chunks)
    assert h.n_duplicates == 4
    assert any(i.id == "duplicates" for i in h.items)


def test_header_only_detected():
    chunks = [_C("X", f"section[{i}]") for i in range(5)]
    h = ch.analyze(chunks)
    assert h.n_header_only == 5
    assert any(i.id == "header_only" for i in h.items)


def test_too_short_detected():
    chunks = [_C("short") for _ in range(5)]
    h = ch.analyze(chunks)
    assert any(i.id == "too_short" for i in h.items)


def test_analyze_accepts_dict_chunks():
    chunks = [{"text": "substantive text " * 5, "structural_path": "section[1]"}]
    h = ch.analyze(chunks)
    assert h.n_chunks == 1


def test_duplicate_structural_numbers_detected():
    # Same number "47" reused by two structurally different nodes — signals a
    # source-corpus numbering collision (e.g. per-file export dropped one of
    # two same-numbered units), independent of duplicate chunk *text*.
    chunks = [
        _C("substantive text about registration " * 5, "document/article[47. Foo]"),
        _C("completely different text about the charter " * 5, "document/article[47]"),
        _C("substantive text of another article " * 5, "document/article[48. Bar]"),
    ]
    h = ch.analyze(chunks)
    assert h.n_duplicate_numbers == 1
    assert any(i.id == "duplicate_structural_numbers" and i.severity == "warn" for i in h.items)


def test_no_duplicate_structural_numbers_when_unique():
    chunks = [_C(f"substantive text of article number {i} " * 3, f"document/article[{i}]") for i in range(5)]
    h = ch.analyze(chunks)
    assert h.n_duplicate_numbers == 0
    assert not any(i.id == "duplicate_structural_numbers" for i in h.items)


# ── completeness (missing structural numbers) ───────────────────────────────

def test_gap_in_sequential_numbering_detected():
    # 1, 2, 3, 5, 6 — "4" is missing from an otherwise sequential corpus.
    # This is the actual signature of a real bug found in production: a
    # malformed source heading meant the parser never captured "4"'s number,
    # so the article was indexed under a numberless leaf — invisible to any
    # check that only looks at duplicate text or duplicate numbers.
    chunks = [_C(f"substantive text of article number {i} " * 3, f"document/article[{i}]")
              for i in (1, 2, 3, 5, 6)]
    h = ch.analyze(chunks)
    assert h.n_missing_numbers == 1
    item = next(i for i in h.items if i.id == "missing_structural_numbers")
    assert "4" in item.detail


def test_no_gap_when_numbering_is_fully_sequential():
    chunks = [_C(f"substantive text of article number {i} " * 3, f"document/article[{i}]") for i in range(1, 8)]
    h = ch.analyze(chunks)
    assert h.n_missing_numbers == 0
    assert not any(i.id == "missing_structural_numbers" for i in h.items)


def test_no_gap_signal_when_corpus_is_not_sequentially_numbered():
    # Sparse, unrelated numbers (e.g. page/footnote refs) spanning a huge
    # range shouldn't drown the report in a "fully missing" false alarm.
    chunks = [_C(f"text {i} " * 3, f"document/article[{i}]") for i in (1, 500, 999)]
    h = ch.analyze(chunks)
    assert h.n_missing_numbers == 0


# ── language mix (sampled, langdetect) ──────────────────────────────────────

# Two closely related languages rather than two arbitrary ones. `langdetect`
# separating Russian from English is easy; separating Russian from Belarusian is
# the case a mixed corpus actually produces, and the one worth pinning. The
# language is the subject here, so these cannot be English; the content is
# neutral and invented.
_RU_TEXT = (
    "Организация ведёт свою деятельность на основании внутреннего регламента, "
    "который определяет порядок работы подразделений и сроки согласования "
    "документов между ними."
)
_BE_TEXT = (
    "Арганізацыя вядзе сваю дзейнасць на падставе ўнутранага рэгламенту, "
    "які вызначае парадак працы падраздзяленняў і тэрміны ўзгаднення "
    "дакументаў паміж імі."
)


def test_single_language_corpus_has_no_mixed_language_item():
    chunks = [_C(_RU_TEXT) for _ in range(20)]
    h = ch.analyze(chunks)
    assert not any(i.id == "mixed_language" for i in h.items)
    assert h.language_sample_size == 20


def test_mixed_language_corpus_flagged():
    chunks = [_C(_RU_TEXT) for _ in range(15)] + [_C(_BE_TEXT) for _ in range(15)]
    h = ch.analyze(chunks)
    assert any(i.id == "mixed_language" for i in h.items)
    assert len(h.language_distribution) >= 1
    assert h.language_sample_size == 30


def test_language_sample_capped_for_large_corpora():
    chunks = [_C(_RU_TEXT) for _ in range(ch._LANGUAGE_SAMPLE_SIZE + 200)]
    h = ch.analyze(chunks)
    assert h.language_sample_size == ch._LANGUAGE_SAMPLE_SIZE


def test_very_short_chunks_excluded_from_language_sample():
    chunks = [_C("12")] * 10  # too short for langdetect, gated by _LANGUAGE_MIN_TEXT_LEN
    h = ch.analyze(chunks)
    assert h.language_sample_size == 0
    assert h.language_distribution == {}


# ── near-duplicate detection (Qdrant k-NN over already-computed vectors) ──────

class _FakeHit:
    def __init__(self, id_, score):
        self.id = id_
        self.score = score


class _FakeQueryResponse:
    def __init__(self, points):
        self.points = points


class _FakeQdrantClient:
    """Maps chunk_id -> list of (neighbor_id, score) the test wants
    query_points(query=chunk_id) to return, self included at score 1.0 —
    mirrors real Qdrant's "query by existing point id" behavior."""

    def __init__(self, neighbors: dict[str, list[tuple[str, float]]]):
        self._neighbors = neighbors

    def query_points(self, collection_name, query, limit, with_payload):
        hits = [_FakeHit(query, 1.0)]
        for neighbor_id, score in self._neighbors.get(query, []):
            hits.append(_FakeHit(neighbor_id, score))
        return _FakeQueryResponse(hits[:limit])


class _FakeQdrantRetriever:
    def __init__(self, neighbors: dict[str, list[tuple[str, float]]]):
        self._client = _FakeQdrantClient(neighbors)
        self._collection = "test_collection"


def test_near_duplicates_detected_above_threshold():
    qdrant = _FakeQdrantRetriever({"a": [("b", 0.99)], "b": [("a", 0.99)]})
    item = ch.detect_near_duplicates(qdrant, ["a", "b"], threshold=0.97)
    assert item is not None
    assert item.id == "near_duplicates"
    assert "1 chunk pair(s)" in item.detail


def test_near_duplicates_none_below_threshold():
    qdrant = _FakeQdrantRetriever({"a": [("b", 0.80)]})
    assert ch.detect_near_duplicates(qdrant, ["a", "b"], threshold=0.97) is None


def test_near_duplicates_deduplicates_symmetric_pairs():
    """a->b and b->a are the same pair — must be counted once, not twice."""
    qdrant = _FakeQdrantRetriever({"a": [("b", 0.99)], "b": [("a", 0.99)]})
    item = ch.detect_near_duplicates(qdrant, ["a", "b"], threshold=0.97)
    assert "1 chunk pair(s)" in item.detail


def test_near_duplicates_empty_sample_returns_none():
    assert ch.detect_near_duplicates(_FakeQdrantRetriever({}), [], threshold=0.97) is None


def test_near_duplicates_survives_a_deleted_point():
    """A point that vanished between sampling and querying (re-indexed,
    deleted) must not crash the whole check — just skip it."""
    class _FlakyClient(_FakeQdrantClient):
        def query_points(self, collection_name, query, limit, with_payload):
            if query == "gone":
                raise Exception("not found")
            return super().query_points(collection_name, query, limit, with_payload)

    qdrant = _FakeQdrantRetriever({"a": [("b", 0.99)]})
    qdrant._client = _FlakyClient({"a": [("b", 0.99)]})
    item = ch.detect_near_duplicates(qdrant, ["gone", "a", "b"], threshold=0.97)
    assert item is not None
