"""End-to-end realm cleanup, kept apart from the fixtures that call it.

A fixture's exit removes its own realm; a fixture's entry removes what an
earlier run left behind by never reaching its exit. The second is not a detail:
an interrupted run does not clean up after itself by definition, and nobody else
will.

A module rather than the body of conftest.py: the cleanup logic has to be
testable, and `conftest` is imported by name and collides with others sharing it.

Both functions take the Mongo module as a parameter rather than importing it
inside. They used to import it, and patching that in a test through `sys.modules`
does not work: with the package already imported, `import adapters.mongodb as
mdb` reads the `adapters.mongodb` attribute rather than the `sys.modules` entry,
so the test quietly talked to the real database. An explicit parameter closes
that hole: the fake has to be passed rather than hoped into place.
"""
from __future__ import annotations

import contextlib
from typing import Any

# Mongo collections the walkthrough writes into. Each one is realm-scoped, so
# teardown can delete by realm_id alone.
REALM_SCOPED_COLLECTIONS = (
    "prompts", "generation_presets", "datasets", "external_rags",
    "corpora", "corpus_ingests", "experiment_runs", "answer_feedback",
    "judgments", "settings",
)

E2E_REALM_PREFIX = "e2e-"


def _real_mongo() -> Any:
    import adapters.mongodb as mdb

    # The module-level Motor client is bound to the loop the app ran on, and
    # that loop is closed by the time teardown runs. Dropping it makes the next
    # get_client() build a fresh one on the loop asyncio.run creates below.
    mdb._client = None
    return mdb


@contextlib.contextmanager
def _fresh_mongo(mongo: Any | None) -> Any:
    """The Mongo module to use, leaving no client bound to a dead loop behind.

    Dropping the client on the way in was already here. Dropping it on the way
    out was not, and that is the half that broke the suite: `asyncio.run` closes
    the loop it made, and the client built inside it stays in the module global,
    bound to that closed loop. Teardown never noticed, because nothing runs
    after teardown.

    The sweep at the start of the session is the caller that did notice. It runs
    before the gateway is built, so the app started up on the dead client and
    every fixture failed with "Event loop is closed" — fifteen errors whose
    message named neither Mongo nor this file.

    An injected `mongo` is a fake belonging to a unit test, so `adapters.mongodb`
    is left alone in that case."""
    if mongo is not None:
        yield mongo
        return
    mdb = _real_mongo()
    try:
        yield mdb
    finally:
        mdb._client = None


async def purge_realm_async(realm_id: str, mdb: Any) -> int:
    """Deletes every Mongo document scoped to a realm, and the realm itself.

    Async so that many realms can be purged inside one event loop. They cannot
    be purged in several: the Motor client binds to the loop it was first used
    on, and a second `asyncio.run` finds it bound to a closed one. Getting that
    wrong is not loud — every operation raises, and a caller that swallows the
    exception reports a number it did not do.
    """
    total = 0
    for collection in REALM_SCOPED_COLLECTIONS:
        for query in ({"realm_id": realm_id}, {"_id": realm_id}):
            for doc in await mdb.find_many(collection, query):
                await mdb.delete_one(collection, {"_id": doc["_id"]})
                total += 1
    for doc in await mdb.find_many("realms", {"id": realm_id}):
        await mdb.delete_one("realms", {"_id": doc["_id"]})
        total += 1
    return total


async def purge_orphans_async(prefix: str, mdb: Any) -> int:
    """Deletes documents whose realm is already gone.

    Deleting the realm first and its documents second is one interruption away
    from leaving exactly this: rows scoped to a realm nothing lists, invisible
    to every screen and to the sweep that goes realm by realm.
    """
    total = 0
    for collection in REALM_SCOPED_COLLECTIONS:
        for doc in await mdb.find_many(collection, {}):
            owner = doc.get("realm_id") or doc.get("_id")
            if isinstance(owner, str) and owner.startswith(prefix):
                await mdb.delete_one(collection, {"_id": doc["_id"]})
                total += 1
    return total


def purge_realm(realm_id: str, mongo: Any | None = None) -> int:
    """Sync wrapper for a single realm — the shape teardown needs."""
    import asyncio

    with _fresh_mongo(mongo) as mdb:
        return asyncio.run(purge_realm_async(realm_id, mdb))


def sweep_stale_realms(keep: str, mongo: Any | None = None) -> list[str]:
    """Removes realms left by an *earlier* run that never reached teardown.

    Teardown cannot clean up after a run that was killed — that is what being
    killed means — so the next run has to. Anything named `e2e-…` other than
    this session's own realm is by definition a leftover: the id is a fresh
    uuid every time.
    """
    import asyncio

    with _fresh_mongo(mongo) as mdb:
        return asyncio.run(_sweep_async(mdb, keep))


def _sweep_async(mdb: Any, keep: str) -> Any:
    async def _sweep() -> list[str]:
        stale = [
            d["id"] for d in await mdb.find_many("realms", {})
            if str(d.get("id", "")).startswith(E2E_REALM_PREFIX) and d.get("id") != keep
        ]
        for rid in stale:
            await purge_realm_async(rid, mdb)
        # Orphaned documents come from a run that managed to delete the realm and
        # not its contents. They cannot be found by realm any more: the realm is
        # gone.
        await purge_orphans_async(E2E_REALM_PREFIX, mdb)
        return stale

    return _sweep()
