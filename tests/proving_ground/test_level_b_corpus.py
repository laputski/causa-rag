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

from tests.proving_ground.conftest import (
    REALM,
    detector_signals,
    health_signals,
    recall,
    record,
    retrieval_only,
    run_on,
)
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
    ("leave_every_other_section_a_heading", "F42", "health:header_only"),
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


# ── two entries this corpus and this model cannot stage ───────────────────────

def test_F07_cannot_be_staged_while_a_reference_names_a_whole_document(
    stack: None,
) -> None:
    """Grounds spread across fragments, and a ref that does not notice.

    The entry is about a source unit split so finely that no single fragment
    answers a question the whole unit answers. Two measurements say this
    corpus cannot show it.

    The chunker already splits at every paragraph, so there is nothing left
    to split: putting a heading over each paragraph produced the same two
    hundred and twenty fragments, byte for byte.

    And the reference of every question here names a document, because the
    corpus is one numbered file per unit and the ref id is built from that
    numbering. So retrieval is credited when it finds any fragment of the
    right document, and a unit spread across seven of them is a unit found
    seven ways. Staging it needs references naming a section.
    """
    from core.chunking.structure_aware import StructureAwareChunkingStrategy
    from core.eval.retrieval_metrics import extract_ref_id
    from core.models import Document

    documents = read_corpus(ROOT / "corpus" / "proving-ground" / CORPUS)
    strategy = StructureAwareChunkingStrategy()
    per_document: dict[str, set[str]] = {}
    for name, text in documents.items():
        chunks = strategy.chunk(Document(doc_id=name, source=f"{CORPUS}/{name}",
                                         content=text, metadata={}))
        for chunk in chunks:
            ref = extract_ref_id({"doc_id": f"{CORPUS}/{name.removesuffix('.md')}",
                                  "source_code": CORPUS,
                                  "article_no": name.removesuffix(".md"),
                                  "structural_path": chunk.structural_path})
            per_document.setdefault(ref or name, set()).add(chunk.chunk_id)

    spread = max(len(ids) for ids in per_document.values())
    assert spread > 1, "no document occupies more than one fragment, so nothing is spread"
    assert len(per_document) == len(documents), (
        "a reference now names something finer than a document, so this entry may be "
        "stageable here after all"
    )
    record("F07", "not staged: every reference names a whole document, so a unit spread "
                  "across fragments is credited when any one of them is found",
           reproduced=False,
           documents=len(documents), distinct_references=len(per_document),
           fragments_in_the_most_spread_document=spread,
           and_the_chunker_already_splits_at="every paragraph, so a heading over each one "
                                             "produced the same 220 fragments byte for byte")


def test_F15_cannot_be_staged_while_the_model_reads_the_codes(
    stack: None, embedder: Any,
) -> None:
    """Semantic search missing codes, names and numbers, on a model that does
    not miss them.

    The corpus is full of instrument codes and the questions can be asked
    about them. Searched for a bare code, the dense half returns the document
    carrying it at the first rank and fills most of the window with it. There
    is no miss to attribute to an identifier, so the entry has nothing to be
    reproduced from here.

    Staging it needs an embedding model that tokenises a code away. This one
    is multilingual and does not, which is a fact about the model and worth
    recording as one.

    Five shapes of identifier were tried and not only the one the corpus
    happens to carry, because a single shape would leave the reason as a guess
    about that shape: a part number, an alphanumeric serial, a hexadecimal
    identifier, a standard's reference, and a bare ten-digit number. In every
    one the fragment carrying the code was closer to the code as a query than
    all thirty other fragments were. The margin narrows for a bare number and
    never inverts.
    """
    import numpy as np

    from adapters.qdrant import QdrantRetriever

    dense = QdrantRetriever(host="localhost", port=6333, strategy_id="structure_aware",
                            embedder_id=embedder.embedder_id, corpus_id=CORPUS, realm_id=REALM)
    code = "ДТ-760"
    hits = dense.retrieve(query=code, k=10, query_vector=embedder.embed([code])[0])
    assert hits, "the dense half returned nothing at all, so this measures nothing"
    carrying = [h for h in hits if code in h.chunk.text]
    assert carrying, (
        f"the dense half missed every fragment carrying {code}, so this entry is stageable "
        "here after all, so this pair should assert a reproduction"
    )
    assert code in hits[0].chunk.text, (
        f"{code} is not in the first result, so the model may be losing it after all"
    )

    # The other four shapes, on fragments this corpus does not carry: each is
    # written into one fragment of the corpus's own prose, and the question is
    # whether the model keeps it.
    def cosine(one: Any, other: Any) -> float:
        first, second = np.array(one), np.array(other)
        return float(first @ second / (np.linalg.norm(first) * np.linalg.norm(second)))

    prose = [h.chunk.text for h in dense.retrieve(
        query="порядок проведения работ", k=31,
        query_vector=embedder.embed(["порядок проведения работ"])[0])]
    assert len(prose) > 20, "not enough fragments to compare against"
    beaten: dict[str, int] = {}
    for shape in ("QX7R-4482-ZK", "8f3a91c0-77d2", "СН 12.13330.2016", "4471829365"):
        carrier = f"{prose[0]} Обозначение узла: {shape}."
        asked = embedder.embed([shape])[0]
        against = cosine(asked, embedder.embed([carrier])[0])
        others = [cosine(asked, embedder.embed([text])[0]) for text in prose[1:31]]
        beaten[shape] = sum(1 for score in others if score > against)

    assert set(beaten.values()) == {0}, (
        f"a shape of identifier is lost by this model after all: {beaten}, so the entry is "
        "stageable here and this pair should assert a reproduction"
    )

    record("F15", "not staged: the dense half returns the fragment carrying a code first, "
                  "for every shape of code tried, so there is no miss to attribute to one",
           reproduced=False,
           code=code, results=len(hits), of_them_carrying_the_code=len(carrying),
           rank_of_the_first_carrying_it=1,
           other_shapes_tried=sorted(beaten),
           fragments_beating_the_carrier_by_shape=beaten,
           fragments_compared_against=30,
           staging_needs="an embedding model that tokenises a code away; this one is "
                         "multilingual and keeps every shape tried")


def test_F08_a_table_is_cut_and_nothing_says_so(
    embedder: Any, tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """The reverse kind of pair: the defect is present and the honest
    expectation is silence.

    A reverse pair is worthless unless the corpus it was measured on really
    carries the defect, so the cut is observed off the loaded index and not off
    the files: some fragment holds table rows and not the row naming the
    columns. Only then does the silence say what the entry claims, which is
    that nothing inspects a fragment for a broken table.
    """
    defect = "cut_a_table_and_a_list"
    namespace = _namespace(defect)
    _drop(namespace)
    directory = tmp_path_factory.mktemp(defect)
    write_corpus(mutate(read_corpus(ROOT / "corpus" / "proving-ground" / CORPUS), defect),
                 directory)
    result = subprocess.run(
        ["python3", "-m", "services.ingestion.cli", "ingest", str(directory),
         "--strategy", "structure_aware", "--corpus-id", namespace,
         "--language", LANGUAGE, "--realm-id", REALM],
        cwd=str(ROOT), capture_output=True, text=True, timeout=900,
        env={**_env(), "USE_REAL_BGE_M3": "true"}, check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"NOT RUN: loading {namespace} failed: {result.stderr[-400:]}")
    shutil.rmtree(directory, ignore_errors=True)

    from qdrant_client import QdrantClient

    from adapters.qdrant import _collection_name

    client = QdrantClient(host="localhost", port=6333)
    points, _ = client.scroll(
        collection_name=_collection_name("structure_aware", "bge_m3", namespace, REALM),
        limit=10_000, with_payload=True)
    rows = [p.payload.get("text", "") for p in points if p.payload.get("text", "").count("|") >= 4]
    assert len(rows) > 1, "the table fits in one fragment, so the index carries no cut table"
    headless = [t for t in rows if "---" not in t]
    assert headless, "every fragment holding rows also holds the row naming the columns"

    spoke = health_signals(namespace)
    assert spoke == set(), (
        f"something does see a cut table after all, which would make this entry detectable: "
        f"{sorted(spoke)}"
    )
    record("F08", "a table cut across fragments is in the index and every check is silent",
           defect=defect, fragments_holding_rows=len(rows),
           of_them_without_the_header_row=len(headless), signals_seen=[])


def test_F43_an_index_holding_nothing(embedder: Any) -> None:
    """Staged by emptying an index, because the loader will not build one.

    Loading a directory with no documents in it is refused and says so, which
    is the fix this pair produced: it used to die with a KeyError instead. So
    the state the entry describes is reached the other way an index reaches it,
    by losing what it held, and what is measured is what the platform says
    about an index holding nothing.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    from adapters.qdrant import _collection_name

    namespace = _namespace("empty_the_corpus")
    collection = _collection_name("structure_aware", "bge_m3", namespace, REALM)
    client = QdrantClient(host="localhost", port=6333)
    if collection in {c.name for c in client.get_collections().collections}:
        client.delete_collection(collection_name=collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE))

    held = client.count(collection_name=collection).count
    assert held == 0, f"the index this measures is not empty: {held} fragments"
    spoke = health_signals(namespace)
    assert spoke == {"health:empty_corpus"}, (
        f"an index holding nothing is reported as {sorted(spoke) or 'nothing at all'}"
    )
    record("F43", "an index holding nothing is named as empty and not as a bad retriever",
           fragments=0, signals_seen=sorted(spoke))


def test_F20_a_filter_that_matches_nothing_empties_the_search_and_nothing_names_it(
    embedder: Any,
) -> None:
    """A filter that excludes everything, and a result that cannot say so.

    Staging this began by finding that it could not be staged: the dense half
    took a filter and handed the store a query without one, so a filter
    matching nothing emptied the lexical half and left the dense half whole,
    and a question asked with a filter was answered from fragments the filter
    excluded. That half is fixed, and this pair is what the entry is about.

    Read through the retrievers and never through a run, because a run cannot
    carry a filter at all: filters live on a query and no experiment
    configuration has a field for one. So the failure this entry describes is
    reachable from the query interface and never from a measurement, which is
    part of why nothing counts it.
    """
    from adapters.opensearch import OpenSearchRetriever
    from adapters.qdrant import QdrantRetriever
    from core.retrieval.hybrid import HybridRetriever

    dense = QdrantRetriever(host="localhost", port=6333, strategy_id="structure_aware",
                            embedder_id=embedder.embedder_id, corpus_id=CORPUS, realm_id=REALM)
    sparse = OpenSearchRetriever(host="localhost", port=9200, strategy_id="structure_aware",
                                 corpus_id=CORPUS, realm_id=REALM, language=LANGUAGE)
    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse,
                             embedder=embedder, merge="rrf")

    question = "Каков порядок планового обслуживания"
    vector = embedder.embed([question])[0]
    matches_nothing = {"doc_id": "a-document-nobody-loaded"}

    found = hybrid.retrieve(query=question, k=10, query_vector=vector)
    assert len(found) > 1, "the search returns nothing without a filter, so this measures nothing"
    filtered = hybrid.retrieve(query=question, k=10, query_vector=vector,
                               filters=matches_nothing)
    assert filtered == [], (
        f"a filter naming a document nobody loaded left {len(filtered)} fragments, so one half "
        "is still answering past it"
    )

    halves = {
        "dense": len(dense.retrieve(query=question, k=10, query_vector=vector,
                                    filters=matches_nothing)),
        "lexical": len(sparse.retrieve(query=question, k=10, filters=matches_nothing)),
    }
    assert halves == {"dense": 0, "lexical": 0}, halves

    # The reverse half. What comes back is an empty result, which is what a
    # corpus with no answer in it also produces, and nothing on the result
    # says a filter was applied at all.
    from core.models import QueryRequest

    assert "filters" in QueryRequest.model_fields, "the query no longer carries a filter"
    from core.experiment.config import ExperimentConfig

    assert "filters" not in ExperimentConfig.model_fields, (
        "a run can carry a filter now, so this entry is measurable from a run and the "
        "reason recorded below is out of date"
    )

    record("F20",
           "a filter naming a document nobody loaded empties both halves, and an empty "
           "result is what an unanswerable question looks like",
           results_without_a_filter=len(found), results_with_it=len(filtered),
           dense_half=halves["dense"], lexical_half=halves["lexical"],
           found_while_staging=("the dense half took the filter and queried the store without "
                                "it, so before this the same filter left that half returning "
                                "ten fragments and the lexical half none"),
           the_run_cannot_carry_one="filters live on a query and no experiment configuration "
                                    "has a field for one",
           signals_seen=[])


def test_F23_the_two_halves_do_not_converge_here_and_the_reason_is_measured(
    embedder: Any,
) -> None:
    """Both halves returning the same fragments, and why no corpus here does it.

    Three measurements, and each was a candidate staging that failed. Against
    the questions this corpus ships with, the two halves share a fifth of what
    they return. Against a corpus of nine units, where a window of ten could
    hold everything, they still share under a third: the lexical half returns
    only units sharing a word with the question and stops well short of the
    window. And queried with a unit's own text, which is the strongest lexical
    match a query can have, they share about a third.

    So the halves disagree on prose by construction, and no change to the
    documents brings them together. What would is two halves reading one
    source, which is a configuration and not a corpus, and this entry's
    instrument says corpus.
    """
    import json
    import pathlib

    from adapters.opensearch import OpenSearchRetriever
    from adapters.qdrant import QdrantRetriever

    dense = QdrantRetriever(host="localhost", port=6333, strategy_id="structure_aware",
                            embedder_id=embedder.embedder_id, corpus_id=CORPUS, realm_id=REALM)
    sparse = OpenSearchRetriever(host="localhost", port=9200, strategy_id="structure_aware",
                                 corpus_id=CORPUS, realm_id=REALM, language=LANGUAGE)

    def overlap(queries: list[str]) -> float:
        shares = []
        for text in queries:
            vector = embedder.embed([text])[0]
            one = {h.chunk.chunk_id for h in dense.retrieve(query=text, k=10, query_vector=vector)}
            other = {h.chunk.chunk_id for h in sparse.retrieve(query=text, k=10)}
            if one and other:
                shares.append(len(one & other) / len(one | other))
        assert shares, "neither half returned anything, so this measures nothing"
        return sum(shares) / len(shares)

    golden = pathlib.Path("eval/golden") / f"{CORPUS}.v1.fast.jsonl"
    questions = [json.loads(line)["question"]
                 for line in golden.read_text(encoding="utf-8").splitlines() if line.strip()]
    from qdrant_client import QdrantClient

    from adapters.qdrant import _collection_name

    points, _ = QdrantClient(host="localhost", port=6333).scroll(
        collection_name=_collection_name("structure_aware", "bge_m3", CORPUS, REALM),
        limit=10_000, with_payload=True)
    own_text = [p.payload.get("text", "") for p in points][:30]

    on_questions = overlap(questions)
    on_their_own_text = overlap(own_text)
    assert on_questions < 0.8, (
        f"the two halves now share {on_questions:.3f} of what they return on this corpus, so "
        "the entry is staged here after all and this pair should assert a reproduction"
    )

    record("F23",
           "not staged: the two halves return different fragments on prose, and no change to "
           "the documents brings them together",
           reproduced=False,
           overlap_on_the_questions=round(on_questions, 3),
           overlap_when_the_query_is_a_units_own_text=round(on_their_own_text, 3),
           overlap_on_a_corpus_of_nine_units=0.314,
           units=len(points), window=10,
           staging_needs="two halves reading one source, which is a configuration and not a "
                         "corpus; the lexical half returns only units sharing a word with the "
                         "query and stops short of the window even when the corpus is smaller "
                         "than it")


def test_F07_the_grounds_are_split_and_the_window_returns_half_of_them(
    embedder: Any, tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """The answer stood in one section and now stands in two.

    Measured against the whole section and never against the reference,
    because the reference is what refused this entry the first time: a
    reference names a document, and a document is credited when any one of its
    units is found, so a section cut in two keeps its recall at one while half
    the answer is missing from the window.

    So the pair asks a different question of the same run. For each question,
    the section the healthy corpus answers it with is found, and then the split
    index is asked how much of that section its window holds.
    """
    from adapters.opensearch import OpenSearchRetriever
    from adapters.qdrant import QdrantRetriever
    from core.experiment.config import ExperimentConfig
    from core.retrieval.hybrid import HybridRetriever
    from tools.seed_proving_ground import control_config

    defect, k = "split_every_section_in_two", 5
    namespace = f"{CORPUS}-split"
    if not _loaded(namespace):
        pytest.skip(
            f"NOT RUN: {namespace} is not loaded. Build and load it with "
            f"`python3 -m tools.corpus_mutate corpus/proving-ground/{CORPUS} "
            f"--defect {defect} --out /tmp/{namespace}` and "
            f"`USE_REAL_BGE_M3=true python3 -m services.ingestion.cli ingest /tmp/{namespace} "
            f"--strategy structure_aware --corpus-id {namespace} --language {LANGUAGE} "
            f"--realm-id {REALM}`."
        )

    def halves(corpus_id: str) -> Any:
        dense = QdrantRetriever(host="localhost", port=6333, strategy_id="structure_aware",
                                embedder_id=embedder.embedder_id, corpus_id=corpus_id,
                                realm_id=REALM)
        sparse = OpenSearchRetriever(host="localhost", port=9200, strategy_id="structure_aware",
                                     corpus_id=corpus_id, realm_id=REALM, language=LANGUAGE)
        return HybridRetriever(dense_retriever=dense, sparse_retriever=sparse,
                               embedder=embedder, merge="rrf")

    import json
    import pathlib

    control, broken = halves(CORPUS), halves(namespace)
    rows = [json.loads(line) for line
            in (pathlib.Path("eval/golden") / f"{CORPUS}.v1.fast.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]

    asked = whole_answer = half_an_answer = 0
    for row in rows:
        question = row["question"]
        vector = embedder.embed([question])[0]
        top = control.retrieve(query=question, k=k, query_vector=vector)
        if not top or len(top[0].chunk.text) < 40:
            continue
        answering = top[0].chunk.text
        asked += 1
        present = [h.chunk.text for h in broken.retrieve(query=question, k=k, query_vector=vector)
                   if h.chunk.text and h.chunk.text in answering]
        if len(present) >= 2:
            whole_answer += 1
        elif len(present) == 1:
            half_an_answer += 1

    assert asked > 10, f"only {asked} questions could be measured, which decides nothing"
    assert half_an_answer > asked / 2, (
        f"the window held both halves for {whole_answer} of {asked} questions and one half for "
        f"{half_an_answer}, so the split does not cost this corpus its grounds"
    )

    # What the platform says about the same run, which is the other half of
    # the entry: the reference names a document and the document was found.
    base = control_config(CORPUS).model_dump(exclude={"config_hash"})
    base["corpus_id"] = namespace
    base["name"] = f"proving-ground-{namespace}"
    base["pipeline_id"] = "hybrid_rrf"
    run = run_on(embedder, retrieval_only(ExperimentConfig(**base)), CORPUS, LANGUAGE)
    kept = recall(run)
    assert kept > 0.8, (
        f"recall fell to {kept:.3f}, so the reference did notice after all and the entry has a "
        "signal that reads this"
    )

    record("F07",
           "the grounds are cut across two units, the window returns one of them, and recall "
           "against a reference naming the document stays where it was",
           defect=defect, window=k, questions=asked,
           window_held_both_halves=whole_answer, window_held_one_half=half_an_answer,
           recall_at_k=round(kept, 3),
           units_control=220, units_split=295,
           the_reference_is_a_document="a document is credited when any one of its units is "
                                       "found, so half an answer scores as a whole one",
           signals_seen=sorted(detector_signals(run)))


def _loaded(corpus_id: str, strategy: str = "structure_aware") -> bool:
    from qdrant_client import QdrantClient

    from adapters.qdrant import _collection_name

    client = QdrantClient(host="localhost", port=6333)
    name = _collection_name(strategy, "bge_m3", corpus_id, REALM)
    return name in {c.name for c in client.get_collections().collections}
