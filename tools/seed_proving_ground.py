"""Seed the proving ground: a realm that is broken on purpose.

The demo realm answers "does the installation work". This one answers a
different question, and the two must not be mixed: the demo has to stay healthy
to be evidence of anything, and deliberately damaged data would destroy exactly
that. So the proving ground is its own realm, and it is marked as one, because
red diagnostics there are the expected outcome and a green proving ground would
mean the guards had stopped firing on the defects put there for them.

What it creates:

    realm     proving-ground            resources pointing at the compose stack
    corpus    base-ru                   10 documents, healthy, the control
    corpus    base-en                   the same ten in English, the control
    dataset   base-ru.v1.fast           15 questions, 13 answerable + 2 refused
    dataset   base-en.v1.fast           the same fifteen in English
    prompt    Grounded handbook QA      active

The two corpora are parallel on purpose. Written from one source, they differ by
language and by nothing else, so a language failure is measured by comparing the
pair instead of by two unrelated observations.

Nothing broken is loaded here. This lays the healthy halves, which every
mutation is measured as a difference from; the damaged corpora come from
`tools.corpus_mutate` and are loaded beside them under their own corpus ids.

Usage:
    python3 -m tools.seed_proving_ground              # idempotent
    python3 -m tools.seed_proving_ground --force      # drop and recreate
    python3 -m tools.seed_proving_ground --no-ingest  # realm and metadata only
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent
GROUND_DIR = REPO_ROOT / "corpus" / "proving-ground"
GOLDEN_DIR = REPO_ROOT / "eval" / "golden"

REALM_ID = "proving-ground"
#: Where the server that answers badly on purpose listens.
#:
#: Written here as well as in that server, because importing it would pull the
#: whole gateway into a seed that has no other need of it. A test asserts the
#: two agree, so the duplication cannot drift into a record pointing at nothing.
FAULTY_RAG_PORT = 8092
STRATEGY = "structure_aware"

# Corpus id, directory, the analyser its index is built with, and a description.
# The analyser matters and is not a detail: an index carries it for life and
# every query inherits it, so a corpus loaded under the wrong one is a defect
# that cannot be undone by any later setting. Naming it here is what makes the
# deliberate mismatch a separate, visible act, never a forgotten flag.
class Corpus(NamedTuple):
    corpus_id: str
    directory: str
    language: str
    description: str


CORPORA: tuple[Corpus, ...] = (
    Corpus(corpus_id="base-ru", directory="base-ru", language="ru_be",
           description="A field service handbook for instruments, in Russian."),
    Corpus(corpus_id="base-en", directory="base-en", language="en",
           description="A field service handbook for instruments, English."),
)

_PROMPT_TEMPLATE = (
    "You answer questions about a field service handbook using ONLY the context "
    "below. When the context does not contain the answer, say so instead of "
    "filling the gap.\n\n"
    # The marker form matters: core/citation.py rewrites "Fragment N" into the
    # real section label of the Nth context entry and recognises only that form.
    "Support every statement with a reference to the context entry it came from, "
    "written as (Fragment N), N being that entry's number in the list below.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


def load_questions(name: str) -> list[dict[str, Any]]:
    path = GOLDEN_DIR / f"{name}.v1.fast.jsonl"
    if not path.exists():
        raise SystemExit(f"Golden set missing: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"Golden set is empty: {path}")
    return rows


# The multilingual reranker. Named explicitly, and this is not a detail: the
# local reranker defaults to an English-only model, so a control configuration
# that said nothing would itself carry the very failure one of the distortions
# stages, and the pair would measure the difference between two broken halves.
CONTROL_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Wider than the selection on purpose. A candidate window equal to the selection
# leaves reranking able to reorder and unable to rescue, which is one of the
# distortions; a control has to start from the other side of that.
CONTROL_TOP_K = 5
CONTROL_FETCH_K = 50


def control_config(corpus_id: str, indexed_as: str = "") -> Any:
    """The healthy configuration every distortion is measured as a change from.

    Hybrid, reranked, with a candidate window wider than the selection: the
    widest shape, so that each distortion has a field to move and refuses for
    its own reason and never for want of one.

    `indexed_as` names a different index to run these same settings against,
    which is what the load-level defects need: they leave the documents and the
    settings alone and put the defect in the index, so the pair is one
    configuration pointed at two namespaces. The question set stays the base
    corpus's, because the questions are the same and only the index differs.
    Without it the pair could not be expressed at all: a distorted namespace is
    not one of the corpora this seed knows, and asking for it was refused.

    Returned, and never stored, because a configuration written down beside
    its distortions drifts away from them the first time somebody edits one.
    """
    from core.experiment.config import ComponentRef, ExperimentConfig

    if corpus_id not in {c.corpus_id for c in CORPORA}:
        raise KeyError(f"Unknown corpus {corpus_id!r}. Known: {[c.corpus_id for c in CORPORA]}")
    target = indexed_as or corpus_id
    if not target.startswith(corpus_id):
        raise ValueError(
            f"{target!r} is not an index of {corpus_id!r}: its questions would be asked of "
            "documents they were not written for, and every retrieval metric would read zero"
        )
    return ExperimentConfig(
        name=f"proving-ground-{target}-control",
        chunking_strategy=ComponentRef(kind="chunker", component_id=STRATEGY),
        embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
        generator=ComponentRef(kind="generator", component_id="ollama"),
        pipeline_id="hybrid_rrf",
        corpus_id=target,
        dataset_name=f"{corpus_id}.v1.fast.jsonl",
        dataset_version="v1",
        top_k=CONTROL_TOP_K,
        fetch_k=CONTROL_FETCH_K,
        merge_strategy="rrf",
        reranker=ComponentRef(kind="reranker", component_id="cross_encoder_local",
                              params={"model_name": CONTROL_RERANKER_MODEL}),
    )


def build_bundle() -> dict[str, Any]:
    """The realm as an import bundle, in the format `export_realm` emits.

    Built and imported through the real import path, so seeding keeps
    exercising the code a person hits when they load a bundle by hand.
    """
    return {
        "format": "causa-realm/v1",
        "exported_at": "2026-01-01T00:00:00+00:00",
        "realm": {
            "id": REALM_ID,
            # English, like the demo realm's. A realm's name and description are
            # stored strings shown verbatim in every locale, so a Russian name
            # reaches an English reader untranslated. The interface labels the
            # realm from `purpose`, which is what carries the meaning in both.
            "name": "Proving Ground",
            # Read by the installation check, which passes this realm over, and
            # by the interface, which marks it. Without the mark its diagnostics
            # read as a broken installation.
            "purpose": "proving_ground",
            "description": (
                "Corpora and settings carrying a defect on purpose. Every guard the "
                "platform has is put to the test of being fooled here, so a red "
                "diagnostic is the expected outcome and not a broken installation."
            ),
            "resources": [
                {"type": "qdrant", "host": "localhost", "port": 6333},
                {"type": "opensearch", "host": "localhost", "port": 9200},
                {"type": "ollama", "host": "localhost", "port": 11434,
                 "model": os.getenv("OLLAMA_MODEL", "qwen3:8b")},
            ],
            "key_metrics": [
                "retrieval_recall_at_k",
                "answer_similarity",
                "context_support",
                "correct_refusal",
            ],
            "created_at": "2026-01-01T00:00:00+00:00",
        },
        # The server that answers badly on purpose, registered here and nowhere
        # else. Its own /health and /capabilities say that it distorts answers
        # and name the mode in force, so a reader who reaches it through this
        # record cannot mistake it for an ordinary system. It is not started by
        # the seed: `make faulty-rag FAULT=<mode>` starts it, one mode per
        # process, because a run is a hundred requests and a mode that could
        # change between them would describe no system anybody operates.
        "external_rags": [{
            "id": "faulty-rag",
            "name": "Faulty RAG (proving ground)",
            "url": f"http://localhost:{FAULTY_RAG_PORT}/",
            "retrieve_endpoint": f"http://localhost:{FAULTY_RAG_PORT}/retrieve",
            "description": (
                "Answers badly on purpose, one named way at a time. Start it with "
                "`make faulty-rag FAULT=<mode>`; `make faulty-rag` alone gives the "
                "control, which answers honestly and is the other half of every pair."
            ),
            "realm_id": REALM_ID,
            "created_at": "2026-01-01T00:00:00+00:00",
        }],
        "prompts": [{
            "id": "proving_ground_prompt_v1",
            "name": "Grounded handbook QA",
            "description": "Answers from the given context only, citing a section for every statement.",
            "version": 1,
            "is_active": True,
            "template": _PROMPT_TEMPLATE,
        }],
        "generation_presets": [],
        "datasets": [
            {
                "filename": f"{corpus_id}.v1.fast.jsonl",
                "name": f"{corpus_id}.v1",
                "version": "v1",
                "speed": "fast",
                "questions": load_questions(corpus_id),
            }
            for corpus_id, _, _, _ in CORPORA
        ],
        "settings": {"active_model": os.getenv("OLLAMA_MODEL", "qwen3:8b"), "active_packs": []},
        "corpora": [
            {"corpus_id": corpus_id, "description": description}
            for corpus_id, _, _, description in CORPORA
        ],
        "masked_fields": [],
    }


async def _drop_realm(realm_id: str) -> int:
    """Removes one realm's documents, and says how many it removed.

    Every query is keyed on `realm_id`, and the only caller passes the module
    constant, so this cannot reach a realm somebody else is using.

    The count is returned because the tolerant `except` below would otherwise
    let `--force` report a clean reseed over documents it never deleted. A
    collection that does not exist yet is genuinely not an error; a delete that
    failed is, and the two are told apart by whether anything was removed."""
    import adapters.mongodb as mdb
    removed = 0
    for collection in ("realms", "prompts", "generation_presets", "datasets", "external_rags", "corpora"):
        try:
            query = {"id": realm_id} if collection == "realms" else {"realm_id": realm_id}
            for doc in await mdb.find_many(collection, query):
                await mdb.delete_one(collection, {"_id": doc["_id"]})
                removed += 1
        except Exception as exc:  # a collection that does not exist yet is not an error
            print(f"  (skipped {collection}: {exc})")
    return removed


async def seed(force: bool = False, do_ingest: bool = True) -> int:
    import adapters.mongodb as mdb
    from services.api_gateway.routers import realms as realms_router
    from services.api_gateway.routers import settings as settings_router

    bundle = build_bundle()
    created: dict[str, str] = {}

    existing = await mdb.find_one("realms", {"id": REALM_ID})
    if existing and force:
        print(f"realm     {REALM_ID}  dropping (--force)")
        removed = await _drop_realm(REALM_ID)
        print(f"  --force: removed {removed} document(s) of realm {REALM_ID!r}")
        existing = None
    if existing:
        created["realm"] = "already present"
    else:
        result = await realms_router.import_realm(bundle, on_conflict="fail")
        created["realm"] = f"created ({result['realm_id']})"

    # import_realm writes settings keyed differently from how settings.py reads
    # them, so the reader's own helper writes them again. Same workaround the
    # demo seed carries, and it goes when that keying is fixed on both sides.
    await settings_router._save_settings(bundle["settings"], REALM_ID)
    created["settings"] = "written"

    if not do_ingest:
        created["corpora"] = "skipped (--no-ingest)"
    else:
        os.environ.setdefault("USE_REAL_BGE_M3", "true")
        from services.api_gateway.routers import corpus as corpus_router
        from services.ingestion.cli import ingest

        for corpus_id, directory, language, description in CORPORA:
            source = GROUND_DIR / directory
            if not source.is_dir():
                raise SystemExit(f"Corpus missing: {source}")
            registered = await mdb.find_one("corpora", {"realm_id": REALM_ID, "corpus_id": corpus_id})
            if registered and (registered.get("backends") or {}) and not force:
                created[f"corpus {corpus_id}"] = "already ingested"
                continue
            print(f"corpus    {corpus_id}  ingesting {len(list(source.glob('*.md')))} documents "
                  f"with the {language!r} analyser…")
            stats = await asyncio.to_thread(
                ingest, source,
                strategy_id=STRATEGY, corpus_id=corpus_id, realm_id=REALM_ID,
                use_opensearch=True, language=language,
            )
            backends: dict[str, dict[str, Any]] = {}
            if stats.get("qdrant_collection"):
                backends["qdrant"] = {"collection": stats["qdrant_collection"], "embedder_id": "bge_m3"}
            if stats.get("opensearch_index"):
                backends["opensearch"] = {"index": stats["opensearch_index"]}
            await corpus_router._register_corpus(
                realm_id=REALM_ID, corpus_id=corpus_id,
                storage_type="hybrid" if "opensearch" in backends else "dense_only",
                backends=backends, owner="platform", description=description,
            )
            created[f"corpus {corpus_id}"] = f"{stats['files']} documents → {stats['chunks']} chunks"

    print()
    print("Proving ground ready. It is meant to be broken; red here is the expected outcome.")
    for key, value in created.items():
        print(f"  {key:20} {value}")
    print()
    print("  Both corpora are healthy: they are the control every mutation is measured against.")
    print("  Damaged copies come from `python3 -m tools.corpus_mutate`.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="drop the realm and recreate it")
    parser.add_argument("--no-ingest", action="store_true", help="realm and metadata without ingesting")
    args = parser.parse_args(argv)
    return asyncio.run(seed(force=args.force, do_ingest=not args.no_ingest))


if __name__ == "__main__":
    raise SystemExit(main())
