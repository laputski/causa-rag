"""Shared fixtures for every test layer.

Two rules live here, and each exists because breaking it is silent.
"""
from __future__ import annotations

import pytest

# The layers that must never load a model. Integration, e2e and the judge
# suites need the real embedder and switch it on themselves.
_STUB_ONLY = ("tests/unit", "tests/contract", "tests/fitness")


@pytest.fixture(autouse=True)
def _stub_embedder_in_the_fast_layers(request, monkeypatch):
    """Keep USE_REAL_BGE_M3 out of the suites that are meant to be fast.

    BgeM3Embedder reads that variable, and deepeval installs itself as a
    pytest plugin that loads `.env`, so a developer whose `.env` sets it
    for running the platform silently gets the real BGE-M3 inside every
    unit test. Measured on this repository: 37 seconds becomes 508, and
    tests written against the deterministic stub start asserting cosine
    thresholds against a real model, which they pass most of the time.

    That is the worst shape a slow suite can take. It is not reproducible
    from the repository alone, since `.env` is not in it, so one machine is
    green and quick while another is slow and occasionally red, and neither
    can show the other why.

    Scoped by path, not applied everywhere: the layers below need
    the real model, and switching it on is the first thing their own
    fixtures do.
    """
    path = str(getattr(request.node, "fspath", ""))
    if any(layer in path.replace("\\", "/") for layer in _STUB_ONLY):
        monkeypatch.delenv("USE_REAL_BGE_M3", raising=False)
    yield


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
