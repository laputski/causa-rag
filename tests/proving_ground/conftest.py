"""Level B: the same failure staged on a live stack, and the same signal read.

Level A puts a defect into a fixture and asks a signal about it. That proves the
signal reads the fixture, and a fixture that has drifted from what the platform
really stores is exactly the claim-from-a-description this project forbids.
Level B removes the fixture: a real corpus in a real index, a real run, and the
signal read off what the run actually recorded.

Both levels are required and neither replaces the other. Level A runs in
continuous integration in under a second and catches a rule that stopped
firing. Level B needs the stack, the embedding model and the loaded indexes,
and catches a rule that fires on a fixture and on nothing else.

**A bait is a pair.** Each test runs the healthy half as well as the broken one
and asserts twice: the signal is silent where the defect is absent, and speaks
where it is present. Half a bait proves nothing, and the half that usually
rots is the silent one, because a rule that fires on everything passes the
loud half every time.

Nothing here loads an index. The loads are printed by
`python3 -m tools.ingest_distort --plan <name>` and run by hand, so the commands
a bait depends on are the commands a person reads. A missing index skips the
test with the command that creates it.
"""
from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
REALM = "proving-ground"
QDRANT = ("localhost", 6333)
OPENSEARCH = ("localhost", 9200)


def _port_open(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def stack() -> None:
    """Skips loudly, because a silent skip reads as a pass.

    Every reason this suite cannot run is named separately: "the stack is down"
    and "the model is not installed" call for different actions from whoever
    reads it.
    """
    missing = [name for name, (host, port) in
               (("qdrant :6333", QDRANT), ("opensearch :9200", OPENSEARCH))
               if not _port_open(host, port)]
    if missing:
        pytest.skip(f"NOT RUN: {', '.join(missing)} unreachable. Bring the stack up first.")
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        pytest.skip("NOT RUN: the real embedding model is not installed, and the stub would "
                    "make every retrieval measurement here meaningless.")


@pytest.fixture(scope="session")
def embedder(stack: None) -> Any:
    """One model for the whole session. Loading it costs seconds; every test
    would otherwise pay them again."""
    os.environ["USE_REAL_BGE_M3"] = "true"
    from adapters.bge_m3 import BgeM3Embedder
    return BgeM3Embedder(use_real_model=True)


class _NeverCalled:
    """A generator that refuses to answer.

    Every run here is retrieval-only, which is what makes a pair affordable: a
    pair of fifteen-question runs costs seconds instead of minutes. If a run
    ever reaches generation, this says so rather than quietly spending an hour.
    """

    generator_id = "never_called"

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a retrieval-only run reached the generator")


def _registry(embedder: Any, corpus_id: str, language: str, strategy: str,
              reranker_model: str = "") -> Any:
    """A registry bound to one index.

    Built per corpus and per analyser, because an OpenSearch index carries its
    analyser for life and the runner's corpus rebind carries the choice
    through: a retriever built with the wrong language would query the right
    index through the wrong stemmer, which is a second defect on top of the one
    under test.
    """
    from adapters.opensearch import OpenSearchRetriever
    from adapters.qdrant import QdrantRetriever
    from core.chunking.fixed import FixedChunkingStrategy
    from core.chunking.structure_aware import StructureAwareChunkingStrategy
    from core.pipeline import NaivePipeline
    from core.registry import ComponentRegistry
    from core.retrieval.hybrid import HybridRetriever

    dense = QdrantRetriever(host=QDRANT[0], port=QDRANT[1], strategy_id=strategy,
                            embedder_id=embedder.embedder_id, corpus_id=corpus_id, realm_id=REALM)
    sparse = OpenSearchRetriever(host=OPENSEARCH[0], port=OPENSEARCH[1], strategy_id=strategy,
                                 corpus_id=corpus_id, realm_id=REALM, language=language)
    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse,
                             embedder=embedder, merge="rrf")
    generator = _NeverCalled()

    registry = ComponentRegistry()
    registry.register("embedder", embedder.embedder_id, embedder)
    registry.register("generator", generator.generator_id, generator)
    registry.register("chunker", FixedChunkingStrategy.strategy_id, FixedChunkingStrategy())
    registry.register("chunker", StructureAwareChunkingStrategy.strategy_id,
                      StructureAwareChunkingStrategy())
    registry.register("pipeline", "naive", NaivePipeline(
        retriever=dense, embedder=embedder, generator=generator, pipeline_id="naive"))
    registry.register("pipeline", "hybrid_rrf", NaivePipeline(
        retriever=hybrid, embedder=embedder, generator=generator, pipeline_id="hybrid_rrf"))
    registry.register("pipeline", "hybrid_weighted", NaivePipeline(
        retriever=HybridRetriever(dense_retriever=dense, sparse_retriever=sparse,
                                  embedder=embedder, merge="weighted"),
        embedder=embedder, generator=generator, pipeline_id="hybrid_weighted"))
    if reranker_model:
        from adapters.reranker import CrossEncoderRerankerLocal
        registry.register("reranker", "cross_encoder_local",
                          CrossEncoderRerankerLocal(model_name=reranker_model))
    return registry


def _index_exists(corpus_id: str, strategy: str, embedder_id: str) -> bool:
    from qdrant_client import QdrantClient

    from adapters.qdrant import _collection_name
    name = _collection_name(strategy, embedder_id, corpus_id, REALM)
    client = QdrantClient(host=QDRANT[0], port=QDRANT[1])
    return name in {c.name for c in client.get_collections().collections}


def run_on(embedder: Any, config: Any, base_corpus: str, language: str,
           strategy: str = "structure_aware", real_model: bool = True) -> dict[str, Any]:
    """One retrieval-only run of the base corpus's questions against `config`.

    `real_model` False queries with the stub while the index may hold real
    vectors or the other way round, which is the only way to stage a model
    mismatch: the mismatch lives between the load and the service, and nothing
    in a configuration can express it.
    """
    from core.experiment.runner import ExperimentRunner
    from eval.dataset import EvalDataset
    from services.api_gateway.routers.experiments import _CompositeEvaluator

    if not _index_exists(config.corpus_id, strategy, embedder.embedder_id):
        pytest.skip(
            f"NOT RUN: no index for corpus {config.corpus_id!r}. Create it with the plan from "
            f"`python3 -m tools.ingest_distort --plan <name> --corpus {base_corpus}`."
        )
    querying = embedder
    if not real_model:
        from adapters.bge_m3 import BgeM3Embedder
        querying = BgeM3Embedder(use_real_model=False)

    registry = _registry(querying, config.corpus_id, language, strategy,
                         reranker_model=(config.reranker.params.get("model_name", "")
                                         if config.reranker else ""))
    dataset = EvalDataset.from_jsonl(ROOT / "eval" / "golden" / f"{base_corpus}.v1.fast.jsonl")
    result = ExperimentRunner(registry).run(
        config, dataset, realm_id=REALM,
        evaluator=_CompositeEvaluator(querying, top_k=config.top_k),
    )
    # The gateway attaches this on the runs it starts, and these do not go
    # through the gateway. Attaching it here is what makes a bait read the
    # same payload a person on the screen would: without it every check that
    # consults the load record would stay silent for a reason having nothing
    # to do with the failure under test.
    result.corpus_manifest = corpus_manifest(config.corpus_id)
    return result.to_dict()


def corpus_manifest(corpus_id: str) -> dict[str, Any]:
    """The load record for one corpus of the proving ground, or nothing.

    Nothing is a real answer, and the callers treat it as one: a corpus
    loaded before manifests existed has no record, and a check comparing a
    run against a record that was never written would compare it with a
    guess.

    A client of its own, and not the module's cached one. The cached client
    binds to the first event loop it sees, and every call here opens a new
    one, so the second read of a session raises "Event loop is closed". The
    first version of this caught that and returned nothing, which is how a
    bait reading the load record passed while reading no record at all: the
    silent fallback is exactly the failure this whole apparatus exists to
    stop, written into the helper whose job was to make a bait honest.
    """
    import asyncio
    import os

    async def _read() -> dict[str, Any]:
        from motor.motor_asyncio import AsyncIOMotorClient

        client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
        try:
            database = client[os.getenv("MONGODB_DB", "ragplatform")]
            doc = await database["corpora"].find_one(
                {"realm_id": REALM, "corpus_id": corpus_id})
            return dict((doc or {}).get("manifest") or {})
        finally:
            client.close()

    return asyncio.run(_read())


def retrieval_only(config: Any) -> Any:
    """The same configuration, stopped before the generator.

    Generation is not what any of these baits read, and paying for it would put
    a pair of runs into the minutes and an hour onto the suite.
    """
    from core.experiment.config import ExperimentConfig

    payload = config.model_dump(exclude={"config_hash"})
    payload["retrieval_only"] = True
    payload["generator"] = {"kind": "generator", "component_id": "never_called", "params": {}}
    return ExperimentConfig(**payload)


def detector_signals(run: dict[str, Any]) -> set[str]:
    from core.eval.detectors import run_detectors
    return {f"detector:{item.id}" for item in run_detectors(run)}


def health_signals(corpus_id: str, strategy: str = "structure_aware") -> set[str]:
    """What the corpus health check says about one loaded index."""
    from qdrant_client import QdrantClient

    from adapters.qdrant import _collection_name
    from core.eval.corpus_health import analyze

    client = QdrantClient(host=QDRANT[0], port=QDRANT[1])
    name = _collection_name(strategy, "bge_m3", corpus_id, REALM)
    points, _ = client.scroll(collection_name=name, limit=10_000, with_payload=True)
    chunks = [{"text": p.payload.get("text", ""),
               "structural_path": p.payload.get("structural_path", ""),
               "doc_id": p.payload.get("doc_id", ""),
               "chunk_id": str(p.id)} for p in points]
    return {f"health:{item.id}" for item in analyze(chunks).items if item.id != "ok"}


def recall(run: dict[str, Any]) -> float:
    return float(run["aggregate_metrics"].get("retrieval_recall_at_k", -1.0))


def recall_before_rerank(run: dict[str, Any]) -> float:
    """What retrieval found, before the reranker had a chance to repair it.

    `retrieval_recall_at_k` on a run that reranks measures the reranker. With a
    candidate window that covers the whole corpus, the cross-encoder reorders
    everything and the vectors stop mattering: a corpus embedded with a stub
    scored 0.46 here and 1.00 there. Any bait about the vectors has to read
    this number, and reading the other one is how such a failure survives
    unnoticed in a system that reranks.
    """
    return float(run["aggregate_metrics"].get("pre_rerank_recall_at_k", -1.0))


def record(failure_id: str, claim: str, **measured: Any) -> None:
    """File what a bait observed, one document per catalogue entry.

    This is what `tools/atlas_report.py` reads to say whether an entry has been
    reproduced on a live stack. The answer used to be a constant returning
    False, which was true on the day it was written and stayed in the file
    after the first entries were reproduced.

    Written by the passing bait and never by hand: a file here is a claim that
    a run was made and that the numbers beside it came out of that run.
    """
    from datetime import UTC, datetime

    out = ROOT / "eval" / "results" / "proving_ground"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{failure_id}.json").write_text(json.dumps({
        "failure_id": failure_id,
        "recorded_at": datetime.now(UTC).isoformat(),
        "claim": claim,
        "measured": measured,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
