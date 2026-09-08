"""Every field of a configuration, and what it does to the run.

Five fields of `ExperimentConfig` have been found accepted and inert, one at
a time, years apart. Each was fixed alone, and the question that would have
found all five at once could not be asked: what the build applies was spread
across a builder and four rebuilding functions, none of which said what it
applied, so there was nothing to compare the configuration against.

`core/experiment/applied.py` says it now, and this is where the saying is
checked. Every field is in exactly one of four tables, and each claim is
observed on a built pipeline:

* said to be applied while building: two configurations differing only in
  it must build differently;
* said to be applied elsewhere: the observation is named, and the pointer is
  resolved against what pytest collects, the same rule the failure catalogue
  follows for its baits;
* said to name the configuration, or said to be applied by nothing: two
  configurations differing only in it must build identically. A claim that
  something changes nothing is a claim, and it is the one this platform is
  least willing to take on trust.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from core.experiment.applied import (
    APPLIED_ELSEWHERE,
    APPLIED_WHILE_BUILDING,
    NAMES_THE_CONFIGURATION,
    NOT_APPLIED,
    describe_built,
)
from core.experiment.config import ComponentRef, ExperimentConfig
from core.experiment.runner import ExperimentRunner
from core.registry import ComponentRegistry

ROOT = Path(__file__).resolve().parents[2]


# ── a machine to build on ────────────────────────────────────────────────────

class _Leaf:
    """A component of any kind at all. What it is never matters here: what is
    observed is which one the build reached for."""

    def __init__(self, name: str = "leaf") -> None:
        self.embedder_id = name
        self.generator_id = name
        self._model_name = name

    def retrieve(self, **kwargs: Any) -> list:
        return []

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1]] * len(texts)

    def generate(self, prompt: str, **params: Any) -> str:
        return ""


class QdrantRetriever:
    """Named for the class the build recognises.

    The build rebuilds a retriever bound to another corpus by looking at the
    name of its class and importing the adapter of that name, so a stand-in
    has to carry the name to be rebound at all, and the adapter itself has to
    be replaced by this one for the rebuild to stay in this process. Both are
    the fixture below. That the build dispatches on a class name is a fragility
    worth seeing written down.
    """

    retriever_id = "qdrant_dense"

    def __init__(self, host: str = "h", port: int = 1, strategy_id: str = "fixed",
                 embedder_id: str = "bge", vector_size: int = 1024,
                 corpus_id: str = "default", realm_id: str | None = None) -> None:
        self._host, self._port = host, port
        self._strategy_id, self._embedder_id = strategy_id, embedder_id
        self._corpus_id, self._realm_id = corpus_id, realm_id

    def retrieve(self, **kwargs: Any) -> list:
        return []


class OpenSearchRetriever(QdrantRetriever):
    retriever_id = "opensearch"

    def __init__(self, host: str = "h", port: int = 2, strategy_id: str = "fixed",
                 corpus_id: str = "default", realm_id: str | None = None,
                 language: str = "ru") -> None:
        super().__init__(host, port, strategy_id, "bge", 1024, corpus_id, realm_id)
        self._language = language


@pytest.fixture(autouse=True)
def _adapters_that_stay_in_this_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real adapters open connections in their constructors, and the
    build constructs one whenever a run names another corpus."""
    monkeypatch.setattr("adapters.qdrant.QdrantRetriever", QdrantRetriever)
    monkeypatch.setattr("adapters.opensearch.OpenSearchRetriever", OpenSearchRetriever)


@pytest.fixture
def runner() -> ExperimentRunner:
    from adapters.ollama_generator import OllamaGenerator
    from core.pipeline import NaivePipeline
    from core.retrieval.graph_hybrid import GraphHybridRetriever
    from core.retrieval.hybrid import HybridRetriever

    registry = ComponentRegistry()
    embedder = _Leaf("bge")
    generator = OllamaGenerator(base_url="http://ollama.test", model="the_default")
    dense = QdrantRetriever()
    sparse = OpenSearchRetriever()

    registry.register("embedder", "bge", embedder)
    registry.register("embedder", "another_model", _Leaf("another_model"))
    registry.register("generator", "ollama", generator)
    registry.register("generator", "another_generator", _Leaf("another_generator"))
    for kind, component_id in (
        ("reranker", "one"), ("reranker", "two"), ("grounder", "a_grounder"),
        ("route_policy", "a_policy"), ("scorer", "a_scorer"),
        ("mask_engine", "a_mask"), ("refusal", "a_refusal"),
    ):
        registry.register(kind, component_id, _Leaf(component_id))

    registry.register("pipeline", "naive", NaivePipeline(
        retriever=dense, embedder=embedder, generator=generator))
    registry.register("pipeline", "hybrid_rrf", NaivePipeline(
        retriever=HybridRetriever(dense_retriever=dense, sparse_retriever=sparse,
                                  embedder=embedder),
        embedder=embedder, generator=generator, pipeline_id="hybrid_rrf"))
    registry.register("pipeline", "graph", NaivePipeline(
        retriever=GraphHybridRetriever(graph_retriever=_Leaf("graph"), base_retriever=dense),
        embedder=embedder, generator=generator, pipeline_id="graph"))

    def _a_registered_system(rag_id: str) -> dict[str, Any]:
        return {"url": f"http://{rag_id}.test", "supported_params": ["fetch_k"]}

    return ExperimentRunner(registry=registry, external_rag_resolver=_a_registered_system)


def _config(**fields: Any) -> ExperimentConfig:
    return ExperimentConfig(**{
        "name": "a configuration",
        "chunking_strategy": ComponentRef(kind="chunker", component_id="fixed"),
        "embedder": ComponentRef(kind="embedder", component_id="bge"),
        "generator": ComponentRef(kind="generator", component_id="ollama"),
        **fields,
    })


def _built(runner: ExperimentRunner, **fields: Any) -> dict[str, Any]:
    return describe_built(runner._build_pipeline(_config(**fields)))


# ── the two values a field is observed between ───────────────────────────────

_A_STEP = {"grounding": "a_grounder", "route_policy": "a_policy", "scorer": "a_scorer",
           "mask_engine": "a_mask", "refusal_policy": "a_refusal"}

TWO_CONFIGURATIONS: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    "pipeline_id": ({"pipeline_id": "naive"}, {"pipeline_id": "hybrid_rrf"}),
    "corpus_id": ({"corpus_id": "default"}, {"corpus_id": "other"}),
    "top_k": ({"top_k": 5}, {"top_k": 9}),
    "fetch_k": ({"fetch_k": None}, {"fetch_k": 40}),
    "merge_strategy": (
        {"pipeline_id": "hybrid_rrf", "merge_strategy": "rrf"},
        {"pipeline_id": "hybrid_rrf", "merge_strategy": "weighted"},
    ),
    "merge_alpha": (
        {"pipeline_id": "hybrid_rrf", "merge_alpha": 0.5},
        {"pipeline_id": "hybrid_rrf", "merge_alpha": 0.9},
    ),
    "rrf_k": (
        {"pipeline_id": "hybrid_rrf", "rrf_k": None},
        {"pipeline_id": "hybrid_rrf", "rrf_k": 10},
    ),
    "graph_weight": (
        {"pipeline_id": "graph", "graph_weight": None},
        {"pipeline_id": "graph", "graph_weight": 0.8},
    ),
    "hops": ({"pipeline_id": "graph", "hops": None}, {"pipeline_id": "graph", "hops": 3}),
    "embedder": (
        {"embedder": ComponentRef(kind="embedder", component_id="bge")},
        {"embedder": ComponentRef(kind="embedder", component_id="another_model")},
    ),
    "generator": (
        {"generator": ComponentRef(kind="generator", component_id="ollama")},
        {"generator": ComponentRef(kind="generator", component_id="another_generator")},
    ),
    "reranker": (
        {"reranker": ComponentRef(kind="reranker", component_id="one")},
        {"reranker": ComponentRef(kind="reranker", component_id="two")},
    ),
    "params": ({"params": {"model": "one_model"}}, {"params": {"model": "another_model"}}),
    "pipeline_source": (
        {"pipeline_source": "in_process", "http_endpoint": "http://a.test"},
        {"pipeline_source": "http", "http_endpoint": "http://a.test"},
    ),
    "http_endpoint": (
        {"pipeline_source": "http", "http_endpoint": "http://one.test"},
        {"pipeline_source": "http", "http_endpoint": "http://two.test"},
    ),
    "external_rag_id": (
        {"pipeline_source": "http", "external_rag_id": "one"},
        {"pipeline_source": "http", "external_rag_id": "two"},
    ),
    **{
        field: ({field: None}, {field: ComponentRef(kind=kind, component_id=component_id)})
        for field, (kind, component_id) in {
            "grounding": ("grounder", "a_grounder"),
            "route_policy": ("route_policy", "a_policy"),
            "scorer": ("scorer", "a_scorer"),
            "mask_engine": ("mask_engine", "a_mask"),
            "refusal_policy": ("refusal", "a_refusal"),
        }.items()
    },
}

# Two values for a field that must change nothing. Kept apart from the pairs
# above because these are the reverse: the observation that passes is the one
# where the two descriptions are the same.
TWO_VALUES_THAT_CHANGE_NOTHING: dict[str, tuple[Any, Any]] = {
    "name": ("one name", "another name"),
    "version": ("1.0.0", "2.0.0"),
    "dataset_version": ("", "v2"),
    "config_schema_version": ("1.0", "1.1"),
    "external_rag_name": (None, "a name for a reader"),
    "seed": (42, 999),
    "chunking_strategy": (
        ComponentRef(kind="chunker", component_id="fixed"),
        ComponentRef(kind="chunker", component_id="sentence"),
    ),
    "retrievers": (
        [],
        [ComponentRef(kind="retriever", component_id="qdrant_dense")],
    ),
}


# ── the partition ────────────────────────────────────────────────────────────

def test_every_field_is_in_exactly_one_table() -> None:
    """The guard the redesign was for. A field added tomorrow fails here until
    somebody says what it does, which is the whole of the mechanism: the trap
    was never a wrong answer, it was no answer at all."""
    tables = {
        "applied while building": set(APPLIED_WHILE_BUILDING),
        "applied elsewhere": set(APPLIED_ELSEWHERE),
        "names the configuration": set(NAMES_THE_CONFIGURATION),
        "applied by nothing": set(NOT_APPLIED),
    }
    fields = set(ExperimentConfig.model_fields)

    spoken_for: set[str] = set()
    twice: list[str] = []
    for named in tables.values():
        twice.extend(sorted(named & spoken_for))
        spoken_for |= named
    assert twice == [], f"fields claimed by two tables at once: {twice}"
    assert sorted(spoken_for - fields) == [], (
        f"tables naming fields the configuration has not got: {sorted(spoken_for - fields)}"
    )
    assert sorted(fields - spoken_for) == [], (
        "configuration fields nobody says anything about: "
        f"{sorted(fields - spoken_for)}"
    )


def test_every_claim_carries_its_reason() -> None:
    """A table of names would be a list; what makes it usable is that each
    entry says what applying the field does, or why nothing does."""
    for table in (APPLIED_WHILE_BUILDING, NAMES_THE_CONFIGURATION, NOT_APPLIED):
        for field, reason in table.items():
            assert len(reason) > 20, f"{field} is claimed without a reason"
    for field, (where, bait) in APPLIED_ELSEWHERE.items():
        assert len(where) > 20, f"{field} is claimed without a reason"
        assert "::" in bait, f"{field} names no observation"


# ── applied while building ───────────────────────────────────────────────────

@pytest.mark.parametrize("field", sorted(APPLIED_WHILE_BUILDING))
def test_a_field_applied_while_building_changes_what_gets_built(
    runner: ExperimentRunner, field: str,
) -> None:
    one, another = TWO_CONFIGURATIONS[field]
    assert one.get(field) != another.get(field), (
        f"the pair for {field} does not differ in {field}"
    )
    assert _built(runner, **one) != _built(runner, **another), (
        f"{field} is claimed to be applied while building and changes nothing there"
    )


def test_the_pairs_cover_every_field_claimed_to_be_applied() -> None:
    """Without this, a field could be claimed applied and silently skipped by
    having no pair, which is the shape of the very defect being guarded."""
    missing = sorted(set(APPLIED_WHILE_BUILDING) - set(TWO_CONFIGURATIONS))
    assert missing == [], f"claimed applied and never observed: {missing}"
    extra = sorted(set(TWO_CONFIGURATIONS) - set(APPLIED_WHILE_BUILDING))
    assert extra == [], f"observed and claimed nowhere: {extra}"

    # The same rule for the reverse claim, minus the one field that cannot be
    # written at all, which has an observation of its own below.
    silent = (set(NAMES_THE_CONFIGURATION) | set(NOT_APPLIED)) - {"config_hash"}
    missing = sorted(silent - set(TWO_VALUES_THAT_CHANGE_NOTHING))
    assert missing == [], f"claimed to change nothing and never observed: {missing}"
    extra = sorted(set(TWO_VALUES_THAT_CHANGE_NOTHING) - silent)
    assert extra == [], f"observed and claimed nowhere: {extra}"


# ── applied elsewhere ────────────────────────────────────────────────────────

def _collected_test_ids(path: Path) -> set[str]:
    """The node identifiers pytest would actually collect from one file."""
    out = subprocess.run(
        [sys.executable, "-m", "pytest", str(path), "--collect-only", "-q", "--no-header",
         "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=ROOT, check=False,
    ).stdout
    return {line.strip() for line in out.splitlines() if "::" in line}


def test_a_field_applied_elsewhere_names_an_observation_that_exists() -> None:
    """Checking that the pointer is filled in checks nothing. The catalogue
    learned this by planting a bait naming a file nobody wrote, and its guard
    accepted it."""
    dangling = []
    for field, (_, bait) in APPLIED_ELSEWHERE.items():
        path, _, node = bait.partition("::")
        collected = _collected_test_ids(ROOT / path)
        assert collected, f"{path} collects no tests at all"
        if bait not in collected:
            dangling.append(f"{field} -> {bait}")
    assert dangling == [], f"observations named and not collectable: {dangling}"


# ── applied by nothing ───────────────────────────────────────────────────────

@pytest.mark.parametrize("field", sorted(set(NAMES_THE_CONFIGURATION) | set(NOT_APPLIED)))
def test_a_field_applied_by_nothing_changes_nothing(
    runner: ExperimentRunner, field: str,
) -> None:
    """The reverse observation, and the one that keeps the last table from
    being a place to park a field nobody wants to think about. Saying a field
    does nothing is a claim, and a claim needs evidence like any other."""
    if field == "config_hash":
        pytest.skip("cannot be set; see the test below")
    one, another = TWO_VALUES_THAT_CHANGE_NOTHING[field]
    assert _built(runner, **{field: one}) == _built(runner, **{field: another}), (
        f"{field} is claimed to change nothing and changes what gets built"
    )


def test_the_fingerprint_cannot_be_written_by_hand() -> None:
    """The one field of the last two tables that cannot be varied: it is
    computed from every other field at construction, so a value handed in is
    overwritten and never obeyed."""
    written = _config(config_hash="a value somebody typed")
    assert written.config_hash != "a value somebody typed"
    assert written.config_hash == _config().config_hash


# ── the two observations the tables above point at ───────────────────────────

def test_a_retrieval_only_run_stops_before_the_generator(runner: ExperimentRunner) -> None:
    """`retrieval_only` is applied while running and not while building, so
    it is observed on the run and not on the pipeline. Nothing observed it
    at all until the tables above demanded a pointer."""
    from eval.dataset import make_stub_dataset

    reached = []

    class _Watched(_Leaf):
        def generate(self, prompt: str, **params: Any) -> str:
            reached.append(prompt)
            return "an answer"

    runner._registry.register("generator", "watched", _Watched("watched"))
    watched = ComponentRef(kind="generator", component_id="watched")
    dataset = make_stub_dataset(n=2)

    runner.run(_config(generator=watched, retrieval_only=True), dataset)
    assert reached == [], "a retrieval-only run reached the generator"

    runner.run(_config(generator=watched), dataset)
    assert reached, "a run that asked for an answer never reached the generator"


def test_a_later_look_at_a_run_loads_the_question_set_it_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`dataset_name` is applied by the services layer and never by the
    build: what it decides is which questions are loaded when somebody comes
    back to a stored run, to widen one question's search or to compare two."""
    import asyncio

    from services.api_gateway.routers import experiments as gateway

    asked_for: list[str] = []

    class _Question:
        question_id = "a_question"

    class _Run:
        dataset_name = "the set this run measured"
        question_results = [_Question()]

    async def _the_run(run_id: str) -> Any:
        return _Run()

    async def _dataset(name: str, external_rag_id: str | None = None) -> Any:
        asked_for.append(name)
        raise AssertionError("stop here: the name is what this observes")

    monkeypatch.setattr(gateway, "_get_one_result", _the_run)
    monkeypatch.setattr(gateway, "_load_dataset", _dataset)

    from fastapi import HTTPException

    with pytest.raises((AssertionError, HTTPException)):
        asyncio.run(gateway.diagnose_retrieval_miss_endpoint("a_run", "a_question"))
    assert asked_for == ["the set this run measured"]
