"""A Realm says which of its resources are somewhere else.

A resource that is not on this machine receives what the Realm's runs send
it: queries, fragments, questions. The Realm is where that is decided, so the
Realm says it, from the addresses alone and without opening a secret.
"""
from __future__ import annotations

from services.api_gateway.routers.realms import _public_realm, resources_elsewhere


def test_a_realm_whose_resources_are_all_here_sends_nothing_away() -> None:
    assert resources_elsewhere([
        {"type": "qdrant", "host": "localhost", "port": 6333},
        {"type": "opensearch", "host": "opensearch", "port": 9200},
        {"type": "neo4j", "uri": "bolt://localhost:7687"},
        {"type": "ollama", "host": "localhost", "port": 11434}]) == []


def test_each_resource_that_is_elsewhere_is_named_with_its_address() -> None:
    elsewhere = resources_elsewhere([
        {"type": "qdrant", "host": "q.example.org", "port": 6333, "https": True, "api_key": {"sealed": "x"}},
        {"type": "opensearch", "host": "localhost", "port": 9200},
        {"type": "neo4j", "uri": "neo4j+s://user:secret@g.example.org:7687"},
        {"type": "openai_compatible", "base_url": "https://api.example.org/v1", "api_key": {"sealed": "y"}}])
    assert elsewhere == [
        {"type": "qdrant", "address": "https://q.example.org:6333"},
        {"type": "neo4j", "address": "neo4j+s://g.example.org:7687"},
        {"type": "openai_compatible", "address": "https://api.example.org"}]


def test_the_realm_the_api_returns_carries_it() -> None:
    realm = _public_realm({"id": "acme", "resources": [
        {"type": "openai_compatible", "base_url": "https://api.example.org/v1"}]})
    assert realm["leaves_this_machine"] == [{"type": "openai_compatible", "address": "https://api.example.org"}]


def test_a_sealed_secret_is_never_opened_to_say_where_a_resource_is() -> None:
    """A sealed value is a mapping. Read as an address it would raise, or be
    printed; neither is asked for here."""
    assert resources_elsewhere([{"type": "qdrant", "host": "localhost", "api_key": {"sealed": "x"}}]) == []
