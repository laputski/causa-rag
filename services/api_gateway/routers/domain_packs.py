"""Domain pack registry — lists packs discoverable under
domain_packs/* (directory-scan), and which are active
(settings.active_packs). Activation takes effect on the next gateway
restart — packs are loaded once in main.py's lifespan(), there is no live
reload mechanism in this stage.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from core.domain.loader import discover_packs
from services.api_gateway.routers.settings import _get_settings_doc, _save_settings

router = APIRouter(prefix="/domain-packs", tags=["domain-packs"])


class ActivatePacksRequest(BaseModel):
    active_packs: list[str]


@router.get("")
async def list_domain_packs(realm_id: str | None = None) -> list[dict[str, Any]]:
    settings = await _get_settings_doc(realm_id)
    active = set(settings.get("active_packs", []))
    return [
        {
            "id": p.id,
            "version": p.version,
            "display_name": p.display_name,
            "description": p.description,
            "exported_kinds": p.exported_kinds,
            "active": p.id in active,
        }
        for p in discover_packs()
    ]


@router.put("")
async def set_active_packs(body: ActivatePacksRequest, realm_id: str | None = None) -> dict[str, Any]:
    settings = await _get_settings_doc(realm_id)
    settings["active_packs"] = body.active_packs
    await _save_settings(settings, realm_id)
    return {"active_packs": body.active_packs, "status": "updated (restart gateway to apply)"}
