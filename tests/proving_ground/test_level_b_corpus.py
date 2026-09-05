"""Paired baits for the failures the documents themselves stage.

The corpus mutator writes a healthy corpus back out with one named defect in
it. What that produces on disk is checked by the unit-level baits; what it
produces in an index is not, and the two differ in everything downstream of the
changed bytes: chunk identifiers, chunk counts, vectors. The health check reads
an index, so the pair has to be built in one.

Unlike the loads of `tools.ingest_distort`, these are performed here. That tool
prints its commands because what it varies is how the load is invoked, and an
analyser or an embedding model chosen behind a reader's back is the very thing
it stages. A corpus defect is about the documents, and the load is incidental
to it.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.proving_ground.conftest import REALM, health_signals, record
from tools.corpus_mutate import mutate, read_corpus, write_corpus

pytestmark = pytest.mark.proving_ground

ROOT = Path(__file__).resolve().parents[2]
CORPUS = "base-ru"
LANGUAGE = "ru_be"

#: Each defect, the catalogue entry it is meant to make observable, and the
#: signal that entry names. Reading it off the catalogue would be better than
#: repeating it here; the entries name several signals each and only one of
#: them speaks about the documents, so which one is the claim of this file.
PAIRS: tuple[tuple[str, str, str], ...] = (
    ("drop_a_numbered_document", "F04", "health:missing_structural_numbers"),
    ("repeat_a_structural_number", "F04", "health:duplicate_structural_numbers"),
    ("shrink_to_fragments", "F06", "health:too_short"),
    ("add_a_second_language", "F14", "health:mixed_language"),
)


def _drop(namespace: str) -> None:
    """Start each pair from an empty index.

    A load adds and never removes, and a mutated corpus is written to a fresh
    temporary directory every run, so the chunk identifiers differ between runs
    and a second run doubles the collection instead of replacing it. Measured:
    the namespaces held between 426 and 880 chunks against the 220 the
    documents produce, and every one of them then reported duplicated text.
    """
    from qdrant_client import QdrantClient

    from adapters.opensearch import _index_name
    from adapters.qdrant import _collection_name

    client = QdrantClient(host="localhost", port=6333)
    collection = _collection_name("structure_aware", "bge_m3", namespace, REALM)
    if collection in {c.name for c in client.get_collections().collections}:
        client.delete_collection(collection_name=collection)
    try:
        from opensearchpy import OpenSearch
        search = OpenSearch(hosts=[{"host": "localhost", "port": 9200}])
        index = _index_name("structure_aware", namespace, REALM)
        if search.indices.exists(index=index):
            search.indices.delete(index=index)
    except Exception:  # the lexical half is optional for these checks
        pass


def _namespace(defect: str) -> str:
    return f"{CORPUS}-{defect.replace('_', '-')}"


@pytest.fixture(scope="module")
def loaded(embedder: Any, tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """One index per defect, built from the corpus this repository ships.

    Loaded once for the module: each load costs seconds, and a pair built
    against an index somebody else's test had already replaced would compare
    two unrelated things.
    """
    healthy = read_corpus(ROOT / "corpus" / "proving-ground" / CORPUS)
    built: dict[str, str] = {}
    for defect, _, _ in PAIRS:
        namespace = _namespace(defect)
        _drop(namespace)
        directory = tmp_path_factory.mktemp(defect)
        write_corpus(mutate(healthy, defect), directory)
        result = subprocess.run(
            ["python3", "-m", "services.ingestion.cli", "ingest", str(directory),
             "--strategy", "structure_aware", "--corpus-id", namespace,
             "--language", LANGUAGE, "--realm-id", REALM],
            cwd=str(ROOT), capture_output=True, text=True, timeout=900,
            env={**_env(), "USE_REAL_BGE_M3": "true"}, check=False,
        )
        if result.returncode != 0:
            pytest.skip(f"NOT RUN: loading {namespace} failed: {result.stderr[-400:]}")
        built[defect] = namespace
        shutil.rmtree(directory, ignore_errors=True)
    return built


def _env() -> dict[str, str]:
    import os
    return dict(os.environ)


def test_the_control_index_holds_what_the_files_hold() -> None:
    """A load adds and never removes, so editing a document leaves the chunk it
    used to produce behind.

    Found by this suite biting its owner: four stale chunks from an earlier
    wording sat in the control index, three of them at a cosine of 1.00 against
    their own replacements, and the healthy corpus therefore read as holding
    near duplicates. Every pair measured against that index was measured
    against a corpus that was clean on disk and stale in the store.

    There is no reload that replaces; dropping the collection and loading again
    is what this asserts has been done.
    """
    from qdrant_client import QdrantClient

    from adapters.qdrant import _collection_name
    from core.chunking.structure_aware import (
        StructureAwareChunkingStrategy,
        tree_from_markdown_headings,
    )
    from core.models import Document

    expected = 0
    for path in sorted((ROOT / "corpus" / "proving-ground" / CORPUS).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        document = Document(source=path.name, content=text, content_hash="h",
                            structure=tree_from_markdown_headings(text))
        expected += len(StructureAwareChunkingStrategy().chunk(document))

    name = _collection_name("structure_aware", "bge_m3", CORPUS, REALM)
    held = QdrantClient(host="localhost", port=6333).count(collection_name=name).count
    assert held == expected, (
        f"the index holds {held} chunks and the documents produce {expected}. Drop the "
        f"collection and load again: `python3 -m tools.seed_proving_ground --force`."
    )


def test_the_healthy_corpus_is_silent() -> None:
    """The premise of every pair below, read off the index the platform ships
    off the index and never off the files: a corpus that is clean on disk and
    damaged in the store would make each pair compare two damaged halves."""
    spoke = health_signals(CORPUS)
    assert spoke == set(), f"the corpus the pairs are measured against is not clean: {sorted(spoke)}"


@pytest.mark.parametrize(("defect", "failure_id", "signal"), PAIRS,
                         ids=[p[0] for p in PAIRS])
def test_a_defect_in_the_documents_reddens_its_signal(
    defect: str, failure_id: str, signal: str, loaded: dict[str, str],
) -> None:
    spoke = health_signals(loaded[defect])
    assert signal in spoke, (
        f"{defect} left nothing for {signal} to see; the index says {sorted(spoke) or 'nothing'}"
    )
    record(failure_id, f"{defect} in the documents is read off the index by {signal}",
           defect=defect, signal=signal, signals_seen=sorted(spoke))


@pytest.mark.parametrize(("defect", "failure_id", "signal"), PAIRS,
                         ids=[p[0] for p in PAIRS])
def test_a_defect_provokes_nothing_it_did_not_name(
    defect: str, failure_id: str, signal: str, loaded: dict[str, str],
) -> None:
    """A pair whose broken half reddens three signals proves something about
    all three and about none in particular. Checked here as well as on the
    files, because a load introduces differences the files do not have."""
    spoke = health_signals(loaded[defect]) - {signal}
    assert spoke == set(), f"{defect} also provoked {sorted(spoke)}"


def test_F26_near_duplicates_are_found_by_the_stored_vectors(
    embedder: Any, tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """The one check that cannot run on a fixture at all.

    Near duplicates are found by asking the vector store for each chunk's own
    neighbours, using the embeddings the load already computed. There is no
    such thing to ask outside a live index, which is why the entry stands at
    "claim unproven" until here.
    """
    from qdrant_client import QdrantClient

    from adapters.qdrant import _collection_name
    from core.eval.corpus_health import detect_near_duplicates

    namespace = _namespace("duplicate_documents")
    _drop(namespace)
    directory = tmp_path_factory.mktemp("dupes")
    write_corpus(mutate(read_corpus(ROOT / "corpus" / "proving-ground" / CORPUS),
                        "duplicate_documents"), directory)
    result = subprocess.run(
        ["python3", "-m", "services.ingestion.cli", "ingest", str(directory),
         "--strategy", "structure_aware", "--corpus-id", namespace,
         "--language", LANGUAGE, "--realm-id", REALM],
        cwd=str(ROOT), capture_output=True, text=True, timeout=900,
        env={**_env(), "USE_REAL_BGE_M3": "true"}, check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"NOT RUN: loading {namespace} failed: {result.stderr[-400:]}")

    client = QdrantClient(host="localhost", port=6333)

    def sample(corpus_id: str) -> tuple[Any, list[str]]:
        name = _collection_name("structure_aware", embedder.embedder_id, corpus_id, REALM)
        points, _ = client.scroll(collection_name=name, limit=200, with_payload=False)
        return _Bound(client, name), [str(p.id) for p in points]

    broken_store, broken_ids = sample(namespace)
    clean_store, clean_ids = sample(CORPUS)

    def pairs(item: Any) -> int:
        return 0 if item is None else int(item.detail.split()[0])

    clean = pairs(detect_near_duplicates(clean_store, clean_ids))
    broken = pairs(detect_near_duplicates(broken_store, broken_ids))

    # Compared as counts and not as presence. The healthy corpus holds one such
    # pair out of two hundred and twenty, because two service cards of the same
    # instrument family differ only by a model code and a figure. That is the
    # case this check exists to surface for a person to judge, and calling it a
    # defect would be reading the check backwards. The corpus holding every
    # document twice holds two hundred and twenty pairs out of four hundred.
    assert clean < len(clean_ids) // 20, (
        f"the healthy corpus reads as {clean} near-duplicate pairs of {len(clean_ids)} chunks, "
        "which is no longer the handful a family of similar documents explains"
    )
    assert broken > len(broken_ids) // 4, (
        f"a corpus holding every document twice reads as only {broken} pairs of {len(broken_ids)}"
    )
    record("F26", "near duplicates are found through the embeddings the load already computed",
           chunks_clean=len(clean_ids), pairs_clean=clean,
           chunks_broken=len(broken_ids), pairs_broken=broken)


class _Bound:
    """A client bound to one collection, which is what the check expects."""

    def __init__(self, client: Any, collection: str) -> None:
        self._client = client
        self._collection = collection

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
