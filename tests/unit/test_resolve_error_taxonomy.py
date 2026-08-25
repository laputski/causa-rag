"""services/api_gateway/routers/feedback.py:_resolve_error_taxonomy — the
generic, pack-agnostic lookup of a feedback-triage error taxonomy from
whichever domain pack(s) are active for a Realm.
Mirrors main.py's own "domain_hooks" resolution; no hardcoded pack id
anywhere in this module (tests/unit/test_p1_guardian.py enforces that
services/ never imports domain_packs directly).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

from services.api_gateway.routers.feedback import _resolve_error_taxonomy

_TAXONOMY = [{"id": "wrong_citation", "definition": "d", "typical_layer": "l", "candidate_levers": []}]


class TestResolveErrorTaxonomy:
    async def test_returns_the_taxonomy_from_an_active_pack_that_provides_one(self) -> None:
        with patch(
            "services.api_gateway.routers.settings._get_settings_doc",
            AsyncMock(return_value={"active_packs": ["some_pack"]}),
        ), patch(
            "core.registry.registry.resolve", return_value={"error_taxonomy": _TAXONOMY},
        ):
            result = await _resolve_error_taxonomy("demo")
        assert result == _TAXONOMY

    async def test_no_active_packs_returns_none(self) -> None:
        with patch(
            "services.api_gateway.routers.settings._get_settings_doc",
            AsyncMock(return_value={"active_packs": []}),
        ):
            result = await _resolve_error_taxonomy("demo")
        assert result is None

    async def test_a_pack_listed_active_but_not_actually_loaded_is_skipped_not_a_crash(self) -> None:
        with patch(
            "services.api_gateway.routers.settings._get_settings_doc",
            AsyncMock(return_value={"active_packs": ["never_loaded"]}),
        ), patch("core.registry.registry.resolve", side_effect=KeyError("not registered")):
            result = await _resolve_error_taxonomy("demo")
        assert result is None

    async def test_an_active_pack_that_provides_no_taxonomy_is_skipped_in_favor_of_the_next(self) -> None:
        def resolve_side_effect(kind, pack_id):
            if pack_id == "pack_without_taxonomy":
                return {"scorer": object()}
            return {"error_taxonomy": _TAXONOMY}

        with patch(
            "services.api_gateway.routers.settings._get_settings_doc",
            AsyncMock(return_value={"active_packs": ["pack_without_taxonomy", "pack_with_taxonomy"]}),
        ), patch("core.registry.registry.resolve", side_effect=resolve_side_effect):
            result = await _resolve_error_taxonomy("demo")
        assert result == _TAXONOMY
