"""Keeping the run store from growing with every run's debugging data.

Runs the rule in `core/run_retention.py` against the store: which runs lose
their retrieval windows, clearing them, and then removing the fragment texts
that no run refers to any more. Once when the gateway starts, every hour
after, and after each saved run.

`tools/prune_run_debug.py` runs the same pass by hand and shows what it would
do before doing it.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog

from core.fragment_texts import text_ids_in
from core.run_retention import Retention, StoredRun, runs_to_clear

log = structlog.get_logger()

#: A text younger than this is left alone even when nothing refers to it: a
#: run being written puts its texts in first and its own documents after.
TEXT_GRACE = timedelta(hours=1)

#: How often the pass runs on its own.
INTERVAL_SECONDS = 3600

_SETTINGS_KEY = "debug_retention"
_lock = asyncio.Lock()


async def retention() -> Retention:
    """The installation's rule, with the defaults for anything not set."""
    import adapters.mongodb as mdb
    try:
        doc = await mdb.find_one("settings", {"_id": "global"}) or {}
    except Exception:
        doc = {}
    stored = doc.get(_SETTINGS_KEY) or {}
    known = asdict(Retention())
    return Retention(**{k: int(stored[k]) for k in known if isinstance(stored.get(k), int)})


async def save_retention(rule: Retention) -> None:
    import adapters.mongodb as mdb
    await mdb.update_one("settings", {"_id": "global"},
                         {"$set": {_SETTINGS_KEY: asdict(rule)}}, upsert=True)


async def _stored_runs() -> tuple[list[StoredRun], set[str]]:
    """Every run the store holds, and those that exist only as a file."""
    import adapters.mongodb as mdb
    from services.api_gateway.routers import experiments as E

    docs = await mdb.find_many("experiment_runs", projection={
        "run_id": 1, "realm_id": 1, "started_at": 1, "debug_cleared_at": 1})
    window_bytes = await mdb.sizes_by(E._WINDOWS_COLLECTION, "run_id")
    runs = [StoredRun(run_id=d.get("run_id", ""), realm_id=d.get("realm_id") or "",
                      started_at=d.get("started_at") or "",
                      window_bytes=window_bytes.get(d.get("run_id", ""), 0),
                      cleared=bool(d.get("debug_cleared_at")))
            for d in docs if d.get("run_id")]
    in_the_database = {r.run_id for r in runs}
    file_only: set[str] = set()
    for path in E._STORE_DIR.glob("*.json"):
        if path.stem in in_the_database:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        file_only.add(path.stem)
        runs.append(StoredRun(run_id=path.stem, realm_id=data.get("realm_id") or "",
                              started_at=data.get("started_at") or "",
                              window_bytes=path.stat().st_size,
                              cleared=bool(data.get("debug_cleared_at"))))
    return runs, file_only


async def _referenced_texts(file_only: set[str]) -> set[str]:
    import adapters.mongodb as mdb
    from services.api_gateway.routers import experiments as E

    fields = E._REF_FIELDS
    keys: set[str] = set()
    runs = await mdb.find_many("experiment_runs", projection={
        f"question_results.{f}.text_id": 1 for f in fields})
    for run in runs:
        keys |= text_ids_in(run.get("question_results") or [], fields)
    windows = await mdb.find_many(E._WINDOWS_COLLECTION,
                                  projection={f"{f}.text_id": 1 for f in fields})
    keys |= text_ids_in(windows, fields)
    for run_id in file_only:
        try:
            data = json.loads((E._STORE_DIR / f"{run_id}.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        keys |= text_ids_in(data.get("question_results") or [], fields)
    return keys


async def _sweep_texts(file_only: set[str], now: datetime, write: bool) -> int:
    """Remove the texts nothing refers to and nothing has touched lately."""
    import adapters.mongodb as mdb
    from services.api_gateway.routers import experiments as E

    cutoff = (now - TEXT_GRACE).isoformat()
    stale = await mdb.find_many(E._TEXTS_COLLECTION, {"seen_at": {"$lt": cutoff}},
                                projection={"_id": 1})
    if not stale:
        return 0
    referenced = await _referenced_texts(file_only)
    orphans = sorted(str(d["_id"]) for d in stale if str(d["_id"]) not in referenced)
    if write:
        for start in range(0, len(orphans), 1000):
            await mdb.delete_many(E._TEXTS_COLLECTION, {"_id": {"$in": orphans[start:start + 1000]}})
    return len(orphans)


async def prune(*, write: bool = True, now: datetime | None = None) -> dict[str, Any]:
    """One pass. Returns what it cleared, or would clear when `write` is False."""
    import adapters.mongodb as mdb
    from services.api_gateway.routers import experiments as E

    now = now or datetime.now(UTC)
    async with _lock:
        rule = await retention()
        runs, file_only = await _stored_runs()
        stored_bytes = (await mdb.data_size(E._WINDOWS_COLLECTION)
                        + await mdb.data_size(E._TEXTS_COLLECTION))
        baseline = await E._get_baseline_run_id()
        chosen = runs_to_clear(runs, rule, stored_bytes=stored_bytes,
                               protected=[baseline] if baseline else [], now=now)
        cleared = []
        for run_id, reason in chosen:
            if not write or await E._clear_windows(run_id, now.isoformat()):
                cleared.append({"run_id": run_id, "reason": reason})
        if write:
            # A run marked and then interrupted before its windows went.
            already = [r.run_id for r in runs if r.cleared]
            if already:
                await mdb.delete_many(E._WINDOWS_COLLECTION, {"run_id": {"$in": already}})
        texts_removed = await _sweep_texts(file_only, now, write)
        report = {
            "rule": asdict(rule), "written": write, "cleared": cleared,
            "texts_removed": texts_removed, "stored_bytes_before": stored_bytes,
        }
        if write:
            report["stored_bytes_after"] = (await mdb.data_size(E._WINDOWS_COLLECTION)
                                            + await mdb.data_size(E._TEXTS_COLLECTION))
    if write and (cleared or texts_removed):
        log.info("run_retention.pruned", **{k: v for k, v in report.items() if k != "rule"})
    return report


async def _safely() -> None:
    try:
        await prune()
    except Exception as exc:
        # Never fatal: the store is only larger than it should be, and the
        # reason is logged where somebody looking for it will find it.
        log.warning("run_retention.failed", error=str(exc))


def enabled() -> bool:
    """Whether the pass runs by itself: only when the installation says so.

    Off unless `CAUSA_RUN_RETENTION` is `on`. Found the hard way: a gateway
    started with `--reload` from a working tree ran the first version of this
    pass the moment its file was saved, and ran a deliberately broken version
    written to test a guard, against the stored runs of that installation.
    The preview in `POST /settings/debug-retention/preview` and the tool
    `tools/prune_run_debug.py` work either way.
    """
    return os.getenv("CAUSA_RUN_RETENTION", "").strip().lower() == "on"


def after_a_run() -> None:
    """Called when a run is saved; the pass itself runs in the background."""
    if not enabled():
        return
    try:
        asyncio.get_running_loop().create_task(_safely())
    except RuntimeError:
        pass


async def keep_pruning() -> None:
    if not enabled():
        log.info("run_retention.off", hint="set CAUSA_RUN_RETENTION=on to clear old runs' windows")
        return
    while True:
        await _safely()
        await asyncio.sleep(INTERVAL_SECONDS)
