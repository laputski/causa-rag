"""`sweep_stale_realms`: cleaning up after a run that never reached its cleanup.

The end-to-end fixture deletes its own realm on exit, but an interrupted run
never reaches that exit: Ctrl-C, a crashed process, a cancelled CI job. Such
realms accumulate in the list beside the real ones — seven of them on a working
installation. Only the next run can remove them, and it does so on entry.

The Mongo fake is passed as a parameter rather than patched into
`sys.modules`: the patch did not take, and the test quietly talked to the real
database. See the module
tests/e2e/realm_cleanup.py.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "e2e" / "realm_cleanup.py"


@pytest.fixture(scope="module")
def cleanup_module():
    # Loaded by path under its own name: `tests/e2e` is not a package on
    # sys.path, and importing `conftest` from there by name collides with every
    # other `conftest`.
    spec = importlib.util.spec_from_file_location("e2e_realm_cleanup", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeMongo:
    """A minimal stand-in for adapters.mongodb: only what the sweep calls."""

    def __init__(self, docs: dict[str, list[dict]]):
        self.docs = docs

    async def find_many(self, collection: str, query: dict) -> list[dict]:
        rows = self.docs.get(collection, [])
        if not query:
            return list(rows)
        return [r for r in rows if all(r.get(k) == v for k, v in query.items())]

    async def delete_one(self, collection: str, query: dict) -> bool:
        rows = self.docs.get(collection, [])
        before = len(rows)
        self.docs[collection] = [
            r for r in rows if not all(r.get(k) == v for k, v in query.items())
        ]
        return len(self.docs[collection]) < before


# @lat: [[realm#Уборка за прерванным прогоном]]
def test_sweep_removes_leftovers_but_never_the_running_session_s_own(cleanup_module) -> None:
    mongo = FakeMongo({
        "realms": [
            {"_id": 1, "id": "acme"},
            {"_id": 2, "id": "e2e-aaaaaaaa"},
            {"_id": 3, "id": "e2e-bbbbbbbb"},
            {"_id": 4, "id": "e2e-current"},
            {"_id": 5, "id": "demo"},
        ],
    })
    removed = cleanup_module.sweep_stale_realms(keep="e2e-current", mongo=mongo)

    assert sorted(removed) == ["e2e-aaaaaaaa", "e2e-bbbbbbbb"]
    # The session's own realm survives the sweep: the session is just starting
    # and needs it.
    assert {r["id"] for r in mongo.docs["realms"]} == {"acme", "e2e-current", "demo"}


# @lat: [[realm#Уборка за прерванным прогоном]]
def test_sweep_leaves_real_realms_alone_when_nothing_is_stale(cleanup_module) -> None:
    # The name is the only signal: a real realm's name means something, a
    # temporary one is a prefix plus a uuid, and the two cannot collide.
    mongo = FakeMongo({"realms": [{"_id": 1, "id": "acme"}, {"_id": 2, "id": "demo"}]})
    assert cleanup_module.sweep_stale_realms(keep="e2e-current", mongo=mongo) == []
    assert len(mongo.docs["realms"]) == 2


def test_purging_a_realm_takes_everything_scoped_to_it(cleanup_module) -> None:
    # A realm is more than its own document: deleting only that leaves runs and
    # datasets orphaned, invisible and undeletable.
    mongo = FakeMongo({
        "realms": [{"_id": 1, "id": "e2e-x"}, {"_id": 2, "id": "acme"}],
        "experiment_runs": [{"_id": 10, "realm_id": "e2e-x"}, {"_id": 11, "realm_id": "acme"}],
        "datasets": [{"_id": 20, "realm_id": "e2e-x"}],
        "settings": [{"_id": "e2e-x"}, {"_id": "acme"}],
    })
    removed = cleanup_module.purge_realm("e2e-x", mongo=mongo)

    assert removed == 4  # the run, the dataset, the settings, the realm itself
    assert [r["_id"] for r in mongo.docs["realms"]] == [2]
    assert [r["_id"] for r in mongo.docs["experiment_runs"]] == [11]
    assert [r["_id"] for r in mongo.docs["settings"]] == ["acme"]
