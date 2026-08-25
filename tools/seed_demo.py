"""Seed the demo realm: a working platform on a machine that has just cloned.

A fresh install lands on an empty shell. There is no realm, so no corpus, no
dataset, no prompt, and nothing to run. The demo closes that gap with one
command and doubles as a live check that realm import, ingestion and corpus
registration all work on this machine.

What it creates:

    realm     demo                      resources pointing at the compose stack
    corpus    handbook                  8 documents from corpus/demo_handbook/
    dataset   handbook.v1.fast          15 questions, 13 answerable + 2 refused
    prompt    Grounded handbook QA      active
    preset    Single-section question   so question generation has a template

Two details decide the shape of this script.

First, ref ids have to match. `core/eval/retrieval_metrics.py#extract_ref_id`
builds "{source_code}/{article_no}", and `services/ingestion/cli.py` derives
those from the parent directory name and a numeric filename stem. So the corpus
must be ingested by the CLI from `corpus/demo_handbook/`, whose files are named
01.md through 08.md, giving stable ids like `demo_handbook/03`. Uploading the
same files through the UI would not work: that path writes to a
`tempfile.mkdtemp(prefix="rag_corpus_")` directory, so `source_code` differs on
every ingest and no pre-baked golden dataset could ever match it.

Second, the dataset lives in `eval/golden/handbook.v1.fast.jsonl` and the bundle
is derived from it, rather than the two being maintained side by side. Keeping
one copy removes the only way they could disagree.

Usage:
    python3 -m tools.seed_demo                  # idempotent, safe to re-run
    python3 -m tools.seed_demo --force          # drop and recreate
    python3 -m tools.seed_demo --no-ingest      # realm and metadata only
    python3 -m tools.seed_demo --write-bundle   # also refresh ui/public/demo.realm.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "corpus" / "demo_handbook"
GOLDEN_FILE = REPO_ROOT / "eval" / "golden" / "handbook.v1.fast.jsonl"
BUNDLE_FILE = REPO_ROOT / "ui" / "public" / "demo.realm.json"

REALM_ID = "demo"
CORPUS_ID = "handbook"
DATASET_FILENAME = "handbook.v1.fast.jsonl"
STRATEGY = "structure_aware"

_PROMPT_TEMPLATE = (
    "You answer questions about an organisation's internal handbook using ONLY "
    "the context below. When the context does not contain the answer, say so "
    "instead of filling the gap.\n\n"
    # The marker form matters: `core/citation.py#substitute_fragment_markers`
    # rewrites "Fragment N" into the real section label of the Nth context
    # entry, and recognises only that form. The demo's first prompt asked for
    # "[section N]", which the substitution passed through untouched, so the
    # answer kept a marker that means nothing without the context list beside
    # it and `computed_citations` came back empty.
    "Support every statement with a reference to the context entry it came "
    "from, written as (Fragment N), N being that entry's number in the list "
    "below. Keep statements drawn from different documents in separate "
    "sentences.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

# A generation preset replaces the built-in template wholesale, so it has to
# ask for the shape the generator parses back: one flat JSON object with
# `question` and `reference_answer`. This one asked for a plain sentence, and
# `_parse_llm_json` found no object in the reply, so every draft failed with
# "unparseable_response", twenty out of twenty, on the demo realm's own
# preset. A preset is the one place an operator can break generation without
# touching code, and the demo shipped with it already broken.
_PRESET_TEMPLATE = (
    "You write control questions for evaluating a RAG system.\n"
    "Below is one excerpt from an organisation's internal handbook. Write ONE "
    "question and its reference answer, using ONLY what the excerpt says. Do "
    "not add facts it does not contain.\n\n"
    "Phrase the question the way an employee would who does not know which "
    "document holds the answer.\n\n"
    "Question type: {question_type}\n{question_type_hint}\n\n"
    "Excerpt:\n{chunk_text}\n\n"
    "Reply with exactly one flat JSON object and nothing else:\n"
    '{"question": "...", "reference_answer": "..."}\n'
    "reference_answer is plain answer text, not JSON and not a nested object."
)


def load_questions() -> list[dict[str, Any]]:
    if not GOLDEN_FILE.exists():
        raise SystemExit(f"Golden dataset missing: {GOLDEN_FILE}")
    rows = [json.loads(line) for line in GOLDEN_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"Golden dataset is empty: {GOLDEN_FILE}")
    return rows


def build_bundle(questions: list[dict[str, Any]]) -> dict[str, Any]:
    """The realm as an import bundle, in the same format `export_realm` emits.

    Deliberately no `neo4j` resource: Neo4j sits behind `profiles: ["graph"]`
    and is not running after a default install, so shipping it would paint a
    red connector on the first screen a new user ever sees.
    """
    return {
        "format": "causa-realm/v1",
        "exported_at": "2026-01-01T00:00:00+00:00",
        "realm": {
            "id": REALM_ID,
            "name": "Demo",
            "description": (
                "A worked example: the internal handbook of a fictional company, "
                "with procedures for purchasing, travel, site access and incidents."
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
                "retrieval_precision_at_k",
            ],
            "created_at": "2026-01-01T00:00:00+00:00",
        },
        "external_rags": [],
        "prompts": [{
            "id": "demo_prompt_v1",
            "name": "Grounded handbook QA",
            "description": "Answers from the given context only, citing a section for every statement.",
            "version": 1,
            "is_active": True,
            "template": _PROMPT_TEMPLATE,
        }],
        "generation_presets": [{
            "id": "demo_preset_single_section",
            "name": "Single-section question",
            "description": "One question answerable from a single excerpt, for checking direct retrieval.",
            "template": _PRESET_TEMPLATE,
        }],
        "datasets": [{
            "filename": DATASET_FILENAME,
            "name": "handbook.v1",
            "version": "v1",
            "speed": "fast",
            "questions": questions,
        }],
        # `active_model` is the key settings.py actually reads. A bundle using
        # "model" here would be silently ignored.
        "settings": {"active_model": os.getenv("OLLAMA_MODEL", "qwen3:8b"), "active_packs": []},
        "corpora": [{"corpus_id": CORPUS_ID, "description": "Operations handbook of a fictional company."}],
        "masked_fields": [],
    }


async def _drop_realm(realm_id: str) -> None:
    import adapters.mongodb as mdb
    for collection in ("realms", "prompts", "generation_presets", "datasets", "external_rags", "corpora"):
        try:
            for doc in await mdb.find_many(collection, {"realm_id": realm_id} if collection != "realms" else {"id": realm_id}):
                await mdb.delete_one(collection, {"_id": doc["_id"]})
        except Exception as exc:  # a collection that does not exist yet is not an error
            print(f"  (skipped {collection}: {exc})")


async def seed(force: bool = False, do_ingest: bool = True, write_bundle: bool = False) -> int:
    import adapters.mongodb as mdb
    from services.api_gateway.routers import realms as realms_router
    from services.api_gateway.routers import settings as settings_router

    questions = load_questions()
    bundle = build_bundle(questions)
    created: dict[str, str] = {}

    # ── realm ────────────────────────────────────────────────────────────────
    existing = await mdb.find_one("realms", {"id": REALM_ID})
    if existing and force:
        print(f"realm     {REALM_ID}  dropping (--force)")
        await _drop_realm(REALM_ID)
        existing = None
    if existing:
        created["realm"] = "already present"
    else:
        # Through the real import path, so the demo keeps exercising the code a
        # user hits when they import a bundle from the welcome screen.
        result = await realms_router.import_realm(bundle, on_conflict="fail")
        created["realm"] = f"created ({result['realm_id']})"

    # ── settings ─────────────────────────────────────────────────────────────
    # import_realm writes settings as {**settings, "realm_id": ...} while
    # settings.py reads them by {"_id": realm_id}, so an imported settings doc
    # is never read back. Writing through the reader's own helper is the fix
    # until that keying mismatch is corrected on both sides.
    await settings_router._save_settings(bundle["settings"], REALM_ID)
    created["settings"] = "written"

    # ── corpus ───────────────────────────────────────────────────────────────
    if not do_ingest:
        created["corpus"] = "skipped (--no-ingest)"
    else:
        registered = await mdb.find_one("corpora", {"realm_id": REALM_ID, "corpus_id": CORPUS_ID})
        if registered and (registered.get("backends") or {}) and not force:
            created["corpus"] = "already ingested"
        else:
            if not CORPUS_DIR.is_dir():
                raise SystemExit(f"Demo corpus missing: {CORPUS_DIR}")
            os.environ.setdefault("USE_REAL_BGE_M3", "true")
            from services.api_gateway.routers import corpus as corpus_router
            from services.ingestion.cli import ingest

            print(f"corpus    {CORPUS_ID}  ingesting {len(list(CORPUS_DIR.glob('*.md')))} documents…")
            stats = await asyncio.to_thread(
                ingest,
                CORPUS_DIR,
                strategy_id=STRATEGY,
                corpus_id=CORPUS_ID,
                realm_id=REALM_ID,
                use_opensearch=True,
            )
            backends: dict[str, dict[str, Any]] = {}
            if stats.get("qdrant_collection"):
                backends["qdrant"] = {"collection": stats["qdrant_collection"], "embedder_id": "bge_m3"}
            if stats.get("opensearch_index"):
                backends["opensearch"] = {"index": stats["opensearch_index"]}
            await corpus_router._register_corpus(
                realm_id=REALM_ID,
                corpus_id=CORPUS_ID,
                storage_type="hybrid" if "opensearch" in backends else "dense_only",
                backends=backends,
                owner="platform",
                description="Operations handbook of a fictional company.",
            )
            created["corpus"] = f"{stats['files']} documents → {stats['chunks']} chunks"

    # ── bundle ───────────────────────────────────────────────────────────────
    if write_bundle:
        BUNDLE_FILE.parent.mkdir(parents=True, exist_ok=True)
        BUNDLE_FILE.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        created["bundle"] = str(BUNDLE_FILE.relative_to(REPO_ROOT))

    answerable = sum(1 for q in questions if q.get("article_refs"))
    print()
    print("Demo realm ready.")
    for key, value in created.items():
        print(f"  {key:9} {value}")
    print(f"  dataset   {DATASET_FILENAME} — {len(questions)} questions "
          f"({answerable} answerable, {len(questions) - answerable} out of scope)")
    print("  prompt    Grounded handbook QA (active)")
    print()
    print("Open the UI, pick the Demo realm and press \"New run\".")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed the demo realm", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="drop the demo realm and recreate it")
    parser.add_argument("--no-ingest", action="store_true", help="create realm metadata without ingesting the corpus")
    parser.add_argument("--write-bundle", action="store_true", help="also refresh ui/public/demo.realm.json from this definition")
    args = parser.parse_args(argv)
    return asyncio.run(seed(force=args.force, do_ingest=not args.no_ingest, write_bundle=args.write_bundle))


if __name__ == "__main__":
    raise SystemExit(main())
