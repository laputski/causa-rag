"""services/api_gateway/main.py:_collect_active_packs_across_realms

Found live: gateway startup used to load only the "global" (no-Realm)
settings doc's active_packs — but the UI never has a "no Realm" state once
any Realm exists, so activating a pack for a real Realm via the Domain
Packs page had no effect on startup, ever (confirmed against the running
instance: the Realm's own settings doc had active_packs=[] while an orphaned
"global" doc named a pack, so /health reported that pack as active while the
Realm's real chat never got any of its hooks).
These pin that startup now unions every Realm's own active_packs.
"""
from __future__ import annotations

from unittest.mock import patch


async def test_unions_active_packs_across_every_realm_plus_global(monkeypatch):
    import services.api_gateway.main as gateway_main

    async def fake_list_realms(include_deleted=False):
        return [{"id": "acme"}, {"id": "acme"}]

    docs = {
        "global": {"active_packs": ["manuals"]},
        "demo": {"active_packs": []},
        "acme": {"active_packs": ["generic_qa"]},
    }

    async def fake_get_settings_doc(realm_id=None):
        return docs[realm_id or "global"]

    monkeypatch.setattr(gateway_main.realms_router, "list_realms", fake_list_realms)
    with patch("services.api_gateway.routers.settings._get_settings_doc", fake_get_settings_doc):
        result = await gateway_main._collect_active_packs_across_realms()

    assert result == ["generic_qa", "manuals"]


async def test_a_realm_with_its_own_activation_is_not_ignored(monkeypatch):
    """The exact bug shape: a Realm's own activation must be picked up even
    when "global" has nothing active at all."""
    import services.api_gateway.main as gateway_main

    async def fake_list_realms(include_deleted=False):
        return [{"id": "demo"}]

    docs = {
        "global": {"active_packs": []},
        "demo": {"active_packs": ["manuals"]},
    }

    async def fake_get_settings_doc(realm_id=None):
        return docs[realm_id or "global"]

    monkeypatch.setattr(gateway_main.realms_router, "list_realms", fake_list_realms)
    with patch("services.api_gateway.routers.settings._get_settings_doc", fake_get_settings_doc):
        result = await gateway_main._collect_active_packs_across_realms()

    assert result == ["manuals"]


async def test_realm_enum_failure_falls_back_to_global_only(monkeypatch):
    import services.api_gateway.main as gateway_main

    async def fake_list_realms(include_deleted=False):
        raise ConnectionError("mongo down")

    async def fake_get_settings_doc(realm_id=None):
        return {"active_packs": ["manuals"]}

    monkeypatch.setattr(gateway_main.realms_router, "list_realms", fake_list_realms)
    with patch("services.api_gateway.routers.settings._get_settings_doc", fake_get_settings_doc):
        result = await gateway_main._collect_active_packs_across_realms()

    assert result == ["manuals"]
