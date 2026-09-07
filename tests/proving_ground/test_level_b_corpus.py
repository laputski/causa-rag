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

def test_F15_the_semantic_half_misses_a_code_the_keyword_half_puts_first(
    embedder: Any,
) -> None:
    """Semantic search missing a designation, and what it took to see it.

    The first attempt asked this corpus and recorded a blocker: searched for a
    bare code, the semantic half returns the fragment carrying it at the first
    rank. Five shapes of designation were tried and the model kept every one,
    so the blocker read as a fact about the model.

    It was a fact about the fragment. A code is a few tokens, and what decides
    a fragment's vector is the rest of it: in a section of a hundred and fifty
    characters the code decides the ranking, at a thousand it is beaten by two
    fragments in twenty-six, and among three thousand fragments of ordinary
    encyclopaedic length it is not in the first fifty. This corpus has two
    hundred and twenty short sections, so it cannot show the failure at all,
    and that is what the blocker was really recording.

    So the pair is measured on the benchmark slice this repository ships, with
    the code written into one passage of it, and the control it needs is the
    small corpus that refused: the same query, the same code, found first.
    """
    from adapters.opensearch import OpenSearchRetriever
    from adapters.qdrant import QdrantRetriever
    from tools.corpus_mutate import THE_CODE_IN_ONE_DOCUMENT

    namespace, strategy = "miracl-ru-coded", "fixed"
    if not _loaded(namespace, strategy):
        pytest.skip(
            f"NOT RUN: {namespace} is not loaded. Build and load it with "
            f"`python3 -m tools.corpus_mutate corpus/miracl-ru "
            f"--defect hide_a_code_in_one_document --out /tmp/{namespace} --limit 1500` and "
            f"`USE_REAL_BGE_M3=true python3 -m services.ingestion.cli ingest /tmp/{namespace} "
            f"--strategy {strategy} --corpus-id {namespace} --language {LANGUAGE} "
            f"--realm-id {REALM}`."
        )

    code = THE_CODE_IN_ONE_DOCUMENT
    asked = embedder.embed([code])[0]

    def ranks(corpus_id: str, of_strategy: str, window: int) -> dict[str, int | None]:
        dense = QdrantRetriever(host="localhost", port=6333, strategy_id=of_strategy,
                                embedder_id=embedder.embedder_id, corpus_id=corpus_id,
                                realm_id=REALM)
        sparse = OpenSearchRetriever(host="localhost", port=9200, strategy_id=of_strategy,
                                     corpus_id=corpus_id, realm_id=REALM, language=LANGUAGE)
        found = dense.retrieve(query=code, k=window, query_vector=asked)
        by_words = sparse.retrieve(query=code, k=window)
        return {
            "semantic": next((i + 1 for i, h in enumerate(found) if code in h.chunk.text), None),
            "keyword": next((i + 1 for i, h in enumerate(by_words) if code in h.chunk.text), None),
            "returned": len(found),
        }

    window = 50
    on_the_slice = ranks(namespace, strategy, window)
    assert on_the_slice["returned"] >= window, (
        f"the semantic half returned {on_the_slice['returned']} fragments for a window of "
        f"{window}, so a miss below it would mean nothing"
    )
    assert on_the_slice["keyword"] == 1, (
        f"the keyword half puts the fragment carrying the code at "
        f"{on_the_slice['keyword']}, so the code is not in the index the way this pair assumes"
    )
    assert on_the_slice["semantic"] is None, (
        f"the semantic half returns it at {on_the_slice['semantic']}, so it does not lose the "
        "code here and this pair should record a blocker again"
    )

    record("F15", "asked for a designation written into one passage, the keyword half returns "
                  "it first and the semantic half does not return it in fifty",
           corpus=namespace, strategy=strategy, code=code, window=window,
           rank_in_the_keyword_half=on_the_slice["keyword"],
           rank_in_the_semantic_half=on_the_slice["semantic"],
           what_the_small_corpus_does="two hundred and twenty sections of about a hundred and "
                                      "thirty characters, where the same query returns the "
                                      "carrying fragment first in both halves",
           what_decides="the length of the fragment and the size of the field it competes in, "
                        "and not the shape of the designation: five shapes were tried")


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


def test_F23_two_halves_reading_one_source_return_one_source(embedder: Any) -> None:
    """Both halves returning the same fragments, and a merge that buys nothing.

    Staging this began by trying to make the corpus do it, and three
    measurements said no corpus here will: against the questions this corpus
    ships with the two halves share a fifth of what they return, against a
    corpus of nine units where a window of ten could hold everything they
    share under a third, and queried with a unit's own text, which is the
    strongest lexical match a query can have, they share about a third. The
    halves disagree on prose by construction.

    What does it is the wiring, and this platform's own classes permit it: the
    merge takes two retrievers and nothing says they must be two, and the
    lexical one accepts a query vector and ignores it, so it can stand on both
    sides. That is the mistake the entry describes, made here on purpose.

    What comes back is the entry exactly. The merged result is what one half
    returns alone, on every question. The trace records both halves, both
    timed, both returning twenty candidates, and nothing compares what they
    returned against each other.
    """
    from adapters.opensearch import OpenSearchRetriever
    from adapters.qdrant import QdrantRetriever
    from core.retrieval.hybrid import HybridRetriever

    dense = QdrantRetriever(host="localhost", port=6333, strategy_id="structure_aware",
                            embedder_id=embedder.embedder_id, corpus_id=CORPUS, realm_id=REALM)
    sparse = OpenSearchRetriever(host="localhost", port=9200, strategy_id="structure_aware",
                                 corpus_id=CORPUS, realm_id=REALM, language=LANGUAGE)
    honest = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse,
                             embedder=embedder, merge="rrf")
    one_source = HybridRetriever(dense_retriever=sparse, sparse_retriever=sparse,
                                 embedder=embedder, merge="rrf")

    import json
    import pathlib

    questions = [json.loads(line)["question"] for line
                 in (pathlib.Path("eval/golden") / f"{CORPUS}.v1.fast.jsonl")
                 .read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(questions) > 10, "too few questions to decide anything"

    def ids(hits: list[Any]) -> list[str]:
        return [h.chunk.chunk_id for h in hits]

    bought_nothing = differed_from_the_honest_one = 0
    trace: dict[str, Any] = {}
    for question in questions:
        vector = embedder.embed([question])[0]
        alone = ids(sparse.retrieve(query=question, k=5))
        merged = ids(one_source.retrieve(query=question, k=5, timings=trace))
        if merged == alone:
            bought_nothing += 1
        if merged != ids(honest.retrieve(query=question, k=5, query_vector=vector)):
            differed_from_the_honest_one += 1

    assert bought_nothing == len(questions), (
        f"the merge changed what one source returned on "
        f"{len(questions) - bought_nothing} questions, so the second retriever is buying "
        "something after all"
    )
    assert differed_from_the_honest_one > len(questions) / 2, (
        "the wiring costs the retrieval nothing on this corpus, so there is no failure here "
        "to be silent about"
    )
    # The trace of the last question, and the shape of the silence: two halves,
    # both timed, both returning candidates, and no word about their being the
    # same candidates.
    assert trace.get("n_dense") and trace.get("n_sparse"), trace
    spoke = detector_signals({"question_results": [], "aggregate_metrics": {},
                              "config": {"pipeline_source": "in_process"}})
    assert "detector:bm25_dominance" not in spoke, spoke

    record("F23",
           "a merge wired to one source returns what that source returns alone, on every "
           "question, and its trace reports two halves both doing work",
           questions=len(questions), merge_returned_one_source_unchanged=bought_nothing,
           differed_from_the_honest_hybrid=differed_from_the_honest_one,
           trace_of_the_last_question=dict(trace),
           the_corpus_cannot_do_this="the two halves share a fifth of what they return on "
                                     "these questions, a third when the query is a unit's own "
                                     "text, and under a third on a corpus of nine units",
           signals_seen=[])


def test_F07_the_grounds_are_split_and_the_window_returns_half_of_them(
    embedder: Any, tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """The answer stood in one section and now stands in two.

    Measured against the whole section and never against the reference,
    because the reference is what refused this entry the first time: a
    reference names a document, and a document is credited when any one of its
    units is found, so a section cut in two keeps its recall at one while half
    the answer is missing from the window. That refusal stood as this entry's
    record until this pair replaced it, and the fact it measured is carried
    below, where it belongs: as the reason the recall says nothing.

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
