"""Rewrite every stored run in the shape today's writer produces.

A run's retrieval windows moved into documents of their own, and a fragment's
text stopped being written once per list. Both happen when a run is written,
so a run written before them keeps the shape it had: on the store this was
written against, seventy-four of a hundred and twenty-nine runs, and two
hundred and ninety-six megabytes where the same runs rewritten are two hundred
and eleven. The largest single run was ten and a quarter megabytes, two thirds
of the engine's own limit for one document, and is two and a half after.

Nothing is written until the run has been taken apart and put back together
and compared with what was stored. That is not ceremony. Run first as a check
alone, it found two ways the reassembly did not return what it was given, one
of which would have destroyed text on a rewrite and could not have been found
any other way:

* a fragment identifier that identifies two different texts, which the
  catalogue has an entry for and one stored run exhibits fifteen times;
* an empty list of windows, written on purpose and taken out with the full
  ones.

Both are fixed in the writer. This tool keeps the check anyway, because the
next thing it finds will be the next thing nobody expected.

A run already in the new shape is read together with its windows before being
taken apart, or the split would find nothing to move and the delete below
would remove the windows it already had.

    python3 -m tools.compact_run_storage           # check every run, write nothing
    python3 -m tools.compact_run_storage --write   # rewrite the ones that would shrink
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any


def _without_the_engines_own_key(document: dict[str, Any]) -> dict[str, Any]:
    """The document as this platform wrote it.

    A replaced document is a new document to the engine and carries a new
    identifier of its own, which is the engine's to choose and not part of
    what a reader of a run sees.
    """
    return {k: v for k, v in document.items() if k != "_id"}


def _size(document: Any) -> int:
    """How much of the store one document is, without the engine's own key.

    Counted without it on both sides of the comparison below, or a run already
    in the new shape looks as though a rewrite would save the length of an
    identifier, and every run is rewritten on every pass.
    """
    return len(json.dumps(_without_the_engines_own_key(document),
                          ensure_ascii=False, default=str).encode())


def _canonical(document: Any) -> str:
    return json.dumps(_without_the_engines_own_key(document),
                      sort_keys=True, ensure_ascii=False, default=str)


async def _compact(write: bool) -> int:
    import adapters.mongodb as mdb
    from services.api_gateway.routers import experiments as store

    runs = await mdb.find_many("experiment_runs", sort=[("started_at", -1)])
    if not runs:
        print("no runs are stored, so there is nothing to rewrite")
        return 0

    refused: list[str] = []
    rewritten = 0
    before = after = 0
    for stored in runs:
        run_id = stored.get("run_id", "")
        windows_it_has = (await store._windows_of([run_id])).get(run_id, [])
        before += _size(stored) + sum(_size(w) for w in windows_it_has)

        # What a reader gets today, which is what must not change: the run
        # document, its windows put back on the questions they belong to, and
        # every reference carrying the text of its fragment again. A run
        # already in the new shape is stored deduplicated, so comparing
        # against the document itself would report every one of them as
        # changed by a rewrite that changes nothing.
        whole = store._restore_texts(store._merge_windows(stored, windows_it_has))

        light, windows = store._split_windows(store._dedupe_texts(whole))
        back = store._restore_texts(store._merge_windows(light, windows))
        if _canonical(back) != _canonical(whole):
            refused.append(run_id)
            after += _size(stored) + sum(_size(w) for w in windows_it_has)
            continue

        after += _size(light) + sum(_size(w) for w in windows)
        saved = (_size(stored) + sum(_size(w) for w in windows_it_has)) - (
            _size(light) + sum(_size(w) for w in windows))
        if saved <= 0:
            continue
        rewritten += 1
        if not write:
            continue
        # Without the engine's own key. A document read out of the store
        # carries it as a string, and writing that string back is refused:
        # the key is the engine's and immutable, and a replacement is a new
        # document to it. The writer never meets this because a run reaches
        # it as a run and not as a document that was already stored.
        await mdb.upsert_one("experiment_runs", {"run_id": run_id},
                             _without_the_engines_own_key(light))
        await mdb.delete_many(store._WINDOWS_COLLECTION, {"run_id": run_id})
        for window in windows:
            await mdb.insert_one(store._WINDOWS_COLLECTION,
                                 _without_the_engines_own_key(window))

        # Read back from the database and compare with what a reader had
        # before. The check above says the shapes are inverse of each other;
        # this says the write actually happened and landed where the reader
        # looks. They are different claims, and a run is worth hours.
        served = store._restore_texts(store._merge_windows(
            await mdb.find_one("experiment_runs", {"run_id": run_id}) or {},
            (await store._windows_of([run_id])).get(run_id, []),
        ))
        if _canonical(served) != _canonical(whole):
            print(f"  STOPPED at {run_id}: what came back is not what a reader had")
            return 2

    verb = "rewritten" if write else "would be rewritten"
    print(f"  {len(runs)} runs stored, {rewritten} {verb}")
    print(f"  {before / 1024 / 1024:.1f} MiB now, {after / 1024 / 1024:.1f} MiB after")
    if refused:
        print(f"  {len(refused)} left alone: taking them apart did not give back what was "
              f"stored, which is a fault in the writer and not in the run")
        for run_id in refused:
            print(f"    {run_id}")
    return 1 if refused else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true",
                        help="rewrite the runs that would shrink; without it nothing is written")
    args = parser.parse_args(argv)
    return asyncio.run(_compact(args.write))


if __name__ == "__main__":
    sys.exit(main())
