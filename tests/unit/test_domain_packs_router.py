"""Domain packs registry router — contract tests (no running Mongo needed),
mirroring tests/unit/test_external_rags_router.py's pattern, plus a real
(no-Mongo, file-fallback) call to list_domain_packs since it doesn't need
write access.
"""
from __future__ import annotations

import asyncio

from services.api_gateway.routers.domain_packs import list_domain_packs, router


def test_router_prefix():
    assert router.prefix == "/domain-packs"


def test_routes_exist():
    paths = {r.path for r in router.routes}
    assert any(p.endswith("/domain-packs") for p in paths)


def test_list_domain_packs_includes_the_packs_that_ship_with_the_platform():
    result = asyncio.run(list_domain_packs())
    ids = {p["id"] for p in result}
    assert "manuals" in ids
    # active defaults to False when MongoDB is unreachable (file-fallback
    # settings have active_packs: [] by default).
    assert all(isinstance(p["active"], bool) for p in result)
