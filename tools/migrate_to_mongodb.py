"""One-shot migration: flat files → MongoDB.

Run once after adding MongoDB to docker-compose:
    python3 -m tools.migrate_to_mongodb
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from motor.motor_asyncio import AsyncIOMotorClient

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGODB_DB", "ragplatform")

ROOT = Path(__file__).parent.parent


async def migrate_prompts(db) -> None:
    prompts_dir = ROOT / "prompts"
    count = 0
    for f in sorted(prompts_dir.glob("*.json")):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
            await db.prompts.replace_one({"id": doc["id"]}, doc, upsert=True)
            count += 1
            print(f"  prompt: {doc['id']} v{doc.get('version', '?')}")
        except Exception as e:
            print(f"  WARN skip {f.name}: {e}")
    print(f"  → {count} prompts migrated")


async def migrate_runs(db) -> None:
    runs_dir = ROOT / "eval" / "results" / "runs"
    count = 0
    for f in sorted(runs_dir.glob("*.json")):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
            run_id = doc.get("run_id", f.stem)
            await db.experiment_runs.replace_one({"run_id": run_id}, doc, upsert=True)
            count += 1
            print(f"  run: {run_id}")
        except Exception as e:
            print(f"  WARN skip {f.name}: {e}")
    print(f"  → {count} runs migrated")


async def migrate_datasets(db) -> None:
    golden_dir = ROOT / "eval" / "golden"
    count = 0
    for f in sorted(golden_dir.glob("*.jsonl")):
        questions = []
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                questions.append(json.loads(line))
        doc = {
            "filename": f.name,
            "name": f.stem,
            "questions": questions,
            "n_questions": len(questions),
        }
        await db.datasets.replace_one({"filename": f.name}, doc, upsert=True)
        count += 1
        print(f"  dataset: {f.name} ({len(questions)} questions)")
    print(f"  → {count} datasets migrated")


async def migrate_deepeval(db) -> None:
    results_dir = ROOT / "eval" / "results"
    count = 0
    for f in sorted(results_dir.glob("deepeval_*.json")):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
            doc["filename"] = f.name
            await db.deepeval_results.replace_one({"filename": f.name}, doc, upsert=True)
            count += 1
            print(f"  deepeval: {f.name}")
        except Exception as e:
            print(f"  WARN skip {f.name}: {e}")
    print(f"  → {count} deepeval results migrated")


async def init_settings(db) -> None:
    existing = await db.settings.find_one({"_id": "global"})
    if not existing:
        await db.settings.insert_one({
            "_id": "global",
            "active_model": os.getenv("OLLAMA_MODEL", "qwen3:8b"),
            "embedder_mode": "real" if os.getenv("USE_REAL_BGE_M3") else "stub",
            "retriever_mode": "hybrid_rrf",
        })
        print("  → settings initialized")
    else:
        print("  → settings already exist, skipped")


async def main() -> None:
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DB_NAME]
    print(f"Connected to MongoDB at {MONGODB_URL}, db={DB_NAME}\n")

    print("=== Migrating prompts ===")
    await migrate_prompts(db)

    print("\n=== Migrating experiment runs ===")
    await migrate_runs(db)

    print("\n=== Migrating datasets ===")
    await migrate_datasets(db)

    print("\n=== Migrating DeepEval results ===")
    await migrate_deepeval(db)

    print("\n=== Initializing settings ===")
    await init_settings(db)

    client.close()
    print("\n✅ Migration complete")


if __name__ == "__main__":
    asyncio.run(main())
