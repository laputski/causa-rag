"""Shared fixtures for every test layer.

One rule lives here, and it exists because breaking it is silent.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _fresh_mongo_client_per_test():
    """Drop the module-level Motor client between tests.

    Motor binds its client to the event loop it is first used on. A test that
    calls `asyncio.run(...)` gets a loop of its own, and `asyncio.run` closes it
    on the way out, leaving a live client in `adapters.mongodb._client` bound to
    a loop that no longer exists. The next test to touch Mongo then fails with
    `RuntimeError: Event loop is closed`, and it fails in a place unrelated to
    whatever left the stale client behind.

    That makes it an ordering bug: every test passes on its own and the suite
    fails, or does not, depending on which packages resolved and in what order
    pytest happened to collect. It surfaced on a freshly resolved virtualenv
    while a months-old one stayed green, which is the worst way for a defect to
    announce itself.

    Resetting between tests costs a client rebuild on next use, which is a
    dictionary lookup and a lazy connection, and buys determinism.
    """
    yield
    try:
        import adapters.mongodb as mdb
    except ImportError:  # a layer that does not use Mongo at all
        return
    mdb._client = None
