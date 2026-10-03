"""Every caller that needs a generator for a Realm gets it from one place.

Five places built one, and they disagreed: two asked the Realm's server and
three the gateway's. The rule they now share is tested here, and the callers
are tested for using it.
"""
from __future__ import annotations

import inspect

import pytest

from services.api_gateway import model_servers as M

_LOCAL = {"host": "models.internal", "port": 11500, "model": "local-model"}
_HOSTED = {"base_url": "https://api.example.org/v1", "api_key": "k", "model": "hosted-model"}


def test_the_one_server_a_realm_declares_is_the_one_used() -> None:
    assert M.generator_for({"ollama": _LOCAL}).served_from()["address"] == "http://models.internal:11500"
    hosted = M.generator_for({"openai_compatible": _HOSTED}).served_from()
    assert (hosted["provider"], hosted["address"], hosted["remote"]) == (
        "openai_compatible", "https://api.example.org/v1", True)


def test_a_realm_with_both_stays_local_unless_a_call_names_the_other() -> None:
    """Adding a hosted server to a Realm must not move every call to it."""
    both = {"ollama": _LOCAL, "openai_compatible": _HOSTED}
    assert M.generator_for(both).served_from()["provider"] == "ollama"
    assert M.generator_for(both, provider="openai_compatible").served_from()["provider"] == "openai_compatible"


def test_a_realm_with_neither_uses_the_gateway_s_local_server(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    assert M.generator_for({}).served_from()["address"] == "http://localhost:11434"
    assert M.generator_for(None).served_from()["provider"] == "ollama"


def test_a_named_model_wins_over_the_resource_s_own() -> None:
    assert M.generator_for({"ollama": _LOCAL}, model="asked-for").served_from()["model"] == "asked-for"
    assert M.generator_for({"ollama": _LOCAL}).served_from()["model"] == "local-model"


def test_the_seed_and_the_window_reach_the_generator() -> None:
    local = M.generator_for({"ollama": _LOCAL}, seed=42, num_ctx=8192)
    assert (local._seed, local._num_ctx) == (42, 8192)
    assert M.generator_for({"openai_compatible": _HOSTED}, seed=42)._seed == 42


@pytest.mark.parametrize("module,function", [
    ("services.api_gateway.main", "_build_realm_scoped_generator"),
    ("services.api_gateway.routers.feedback", "_resolve_triage_generator"),
    ("services.api_gateway.routers.generation", "_run_generation_background"),
    ("services.api_gateway.routers.prompts", "generate_prompt_draft"),
    ("services.api_gateway.routers.experiments", "_resolve_judge"),
])
def test_every_place_that_needs_a_generator_asks_here(module: str, function: str) -> None:
    """Read from the source, because the defect was a caller building its own:
    a new one doing the same would pass every behavioural test written for
    the others."""
    import importlib

    source = inspect.getsource(getattr(importlib.import_module(module), function))
    assert "OllamaGenerator(" not in source and "OpenAICompatibleGenerator(" not in source, (
        f"{function} builds a generator itself")
    assert "generator_for" in source, f"{function} does not ask services/api_gateway/model_servers.py"
