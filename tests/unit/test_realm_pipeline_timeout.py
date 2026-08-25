"""_resolve_realm_pipeline (chat routing) must honor a registered RAG's
timeout_s the same way the experiment runner and the /test probe do —
HttpPipeline's flat 30s default is too short for some RAGs (found live
registering a multi-step agentic RAG)."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from services.api_gateway.main import _resolve_realm_pipeline


@pytest.mark.asyncio
async def test_explicit_external_rag_id_uses_registered_timeout_s():
    doc = {"id": "abc123", "url": "http://slow.example/rag", "realm_id": "acme", "timeout_s": 180.0}
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=doc)):
        pipeline = await _resolve_realm_pipeline("acme", external_rag_id="abc123")
    assert pipeline._timeout == 180.0


@pytest.mark.asyncio
async def test_explicit_external_rag_id_defaults_to_30s_when_unset():
    doc = {"id": "abc123", "url": "http://fast.example/rag", "realm_id": "acme"}
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=doc)):
        pipeline = await _resolve_realm_pipeline("acme", external_rag_id="abc123")
    assert pipeline._timeout == 30.0


@pytest.mark.asyncio
async def test_auto_pick_single_rag_uses_registered_timeout_s():
    docs = [{"id": "abc123", "url": "http://slow.example/rag", "realm_id": "acme", "timeout_s": 90.0}]
    with patch("adapters.mongodb.find_many", AsyncMock(return_value=docs)):
        pipeline = await _resolve_realm_pipeline("acme")
    assert pipeline._timeout == 90.0
