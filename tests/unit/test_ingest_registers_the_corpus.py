"""A corpus loaded into a realm has to reach that realm's own list.

The loading command takes a realm identifier and, until now, the realm learned
nothing from it: the index existed, every metric could be computed against it,
and it appeared in no list on any screen. Found while writing a manual check of
the proving ground, where a deliberately damaged corpus was loaded and then
could not be selected to look at, which is half of what that realm is for.

The registry record is what the interface lists, and the helper that writes it
declares itself idempotent and safe to call on every successful load. The
gateway's own ingest route has always called it; the command line had not.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

from services.ingestion.cli import _register_in_the_realm

RESULT = {
    "files": 10, "chunks": 220,
    "qdrant_collection": "proving-ground__base-ru__structure_aware__bge_m3",
    "opensearch_index": "rag__proving-ground__base-ru__structure_aware",
}


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {}


def _register(result: dict[str, Any] = RESULT) -> tuple[_Recorder, str]:
    recorder = _Recorder()
    with patch("services.api_gateway.routers.corpus._register_corpus", recorder):
        said = _register_in_the_realm("proving-ground", "base-ru", result)
    return recorder, said


def test_the_realm_is_told_which_corpus_was_loaded() -> None:
    recorder, said = _register()
    assert len(recorder.calls) == 1
    assert recorder.calls[0]["realm_id"] == "proving-ground"
    assert recorder.calls[0]["corpus_id"] == "base-ru"
    assert "base-ru" in said and "proving-ground" in said


def test_both_indexes_are_recorded_by_the_names_the_load_actually_used() -> None:
    """Derived here would mean deriving the collection name a second time, and
    a second derivation drifts from the first. The loader returns what it
    wrote."""
    backends = _register()[0].calls[0]["backends"]
    assert backends["qdrant"]["collection"] == RESULT["qdrant_collection"]
    assert backends["opensearch"]["index"] == RESULT["opensearch_index"]


def test_a_load_without_the_lexical_half_is_recorded_as_dense_only() -> None:
    """One of the proving ground's own distortions loads the semantic half
    alone. Recording it as hybrid would describe an index that is not there."""
    call = _register({**RESULT, "opensearch_index": None})[0].calls[0]
    assert call["storage_type"] == "dense_only"
    assert "opensearch" not in call["backends"]


def test_a_load_with_both_halves_is_recorded_as_hybrid() -> None:
    assert _register()[0].calls[0]["storage_type"] == "hybrid"


def test_a_registry_that_refuses_does_not_fail_the_load() -> None:
    """The documents are in the index either way, and a load that succeeded is
    not undone by a database that was unreachable a second later. What must not
    happen is silence: the failure is returned for the caller to print."""
    async def refuse(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("mongodb unreachable")

    with patch("services.api_gateway.routers.corpus._register_corpus", refuse):
        said = _register_in_the_realm("proving-ground", "base-ru", RESULT)
    assert "not registered" in said
    assert "mongodb unreachable" in said


def test_the_command_line_calls_it_only_when_a_realm_was_named() -> None:
    """A load with no realm belongs to no realm's list. Read out of the source,
    because reaching this branch end to end needs a whole stack up."""
    import pathlib

    source = (pathlib.Path(__file__).parents[2] / "services" / "ingestion" / "cli.py").read_text(
        encoding="utf-8")
    assert "if args.realm_id:" in source
    assert "_register_in_the_realm(args.realm_id" in source


# ── the manifest ──────────────────────────────────────────────────────────────

def test_the_manifest_records_whether_a_model_or_a_hash_made_the_vectors() -> None:
    """The sharpest thing in it, and the only place the answer exists.

    The embedder's id and version are class constants, identical for the
    working model and for the stub that hashes text into a vector, so a
    corpus indexed by either is described the same way everywhere
    afterwards. Asking at load time is not a convenience; it is the only
    moment anybody can tell.
    """
    from adapters.bge_m3 import BgeM3Embedder
    from services.ingestion.cli import _manifest

    stub = _manifest(BgeM3Embedder(use_real_model=False), "fixed", 800, 100, ["a", "b"], 12)
    assert stub["embedder_is_real_model"] is False
    assert stub["embedder_id"] == "bge_m3"
    assert stub["embedder_version"] == "1.0.0"


def test_the_digest_is_a_property_of_the_set_and_not_of_the_walk() -> None:
    """Two loads of the same documents in a different file order describe
    one corpus, and a digest that said otherwise would report a corpus
    changed every time the directory was read in another order."""
    from adapters.bge_m3 import BgeM3Embedder
    from services.ingestion.cli import _manifest

    embedder = BgeM3Embedder(use_real_model=False)
    one = _manifest(embedder, "fixed", 800, 100, ["h1", "h2", "h3"], 9)
    other = _manifest(embedder, "fixed", 800, 100, ["h3", "h1", "h2"], 9)
    assert one["documents_digest"] == other["documents_digest"]


def test_a_changed_document_changes_the_digest() -> None:
    from adapters.bge_m3 import BgeM3Embedder
    from services.ingestion.cli import _manifest

    embedder = BgeM3Embedder(use_real_model=False)
    before = _manifest(embedder, "fixed", 800, 100, ["h1", "h2"], 6)
    after = _manifest(embedder, "fixed", 800, 100, ["h1", "h2-edited"], 6)
    assert before["documents_digest"] != after["documents_digest"]


def test_the_registry_record_reads_the_embedder_off_what_indexed() -> None:
    """It used to be the literal "bge_m3", true of every load anybody had
    run and recorded regardless of what indexed the next one."""
    from pathlib import Path as _Path

    source = _Path(__file__).resolve().parents[2] / "services" / "ingestion" / "cli.py"
    text = source.read_text(encoding="utf-8")
    assert '"embedder_id": "bge_m3"' not in text
    assert 'manifest.get("embedder_id"' in text


def test_the_manifest_carries_the_strategy_s_own_verdict_on_its_output() -> None:
    """A strategy named for structure that produced none has done exactly
    what the plain fixed-window strategy does, under another name, and every
    other trace of that load looks correct."""
    from adapters.bge_m3 import BgeM3Embedder
    from core.chunking.structure_aware import StructureAwareChunkingStrategy
    from core.models import Document
    from services.ingestion.cli import _manifest

    flat = Document(doc_id="d", source="d.md",
                    content="Prose carrying no heading at all. " * 60, metadata={})
    chunks = StructureAwareChunkingStrategy().chunk(flat)
    manifest = _manifest(BgeM3Embedder(use_real_model=False), "structure_aware", 512, 64,
                         ["h"], len(chunks), chunks)
    assert manifest["post_conditions_unmet"], manifest


def test_a_structured_corpus_keeps_the_promise() -> None:
    """The half that rots. A check reporting a broken promise on every load
    would say nothing about any of them."""
    from adapters.bge_m3 import BgeM3Embedder
    from core.chunking.structure_aware import StructureAwareChunkingStrategy
    from core.models import Document
    from services.ingestion.cli import _manifest

    structured = Document(
        doc_id="d", source="d.md", metadata={},
        content="# One\n\nProse under the first heading. " * 10
                + "\n\n# Two\n\nProse under the second heading. " * 10,
    )
    chunks = StructureAwareChunkingStrategy().chunk(structured)
    manifest = _manifest(BgeM3Embedder(use_real_model=False), "structure_aware", 512, 64,
                         ["h"], len(chunks), chunks)
    assert manifest["post_conditions_unmet"] == [], manifest
