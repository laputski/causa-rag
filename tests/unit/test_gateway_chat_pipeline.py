"""services/api_gateway/main.py:_build_chat_pipeline — refusal_policy/grounder wiring,
and (since Realm parameterization) retriever/generator resolution per Realm.

Locks in a real bug: a domain pack's refusal_policy hook looked connected
in chat (_build_chat_pipeline passed it to ConfigurablePipeline) but could
never actually fire — core/pipeline.py only consults refusal_policy when
grounding already ran (`if self._refusal_policy is not None and gr is not
None`), and _build_chat_pipeline never wired a grounder at all. The
low-confidence fallback was silently dead in the live chat path while
working fine in experiment runs (core/experiment/runner.py does wire a
grounder from config.grounding).

_build_chat_pipeline became async and resolves its base retriever/generator
per Realm/corpus_id (_build_realm_scoped_retriever/_generator) instead of
always reusing the registry's single startup-time instances — see
the design notes "Chat routing via Realm". These tests stub those two out
(they'd otherwise attempt a real Qdrant/Ollama connection) and register a
real embedder id, since the new code path resolves "embedder"/"bge_m3" from
the registry directly rather than pulling it off a pre-built pipeline object.
"""
from __future__ import annotations

from core.refusal import ThresholdRefusalPolicy
from core.registry import ComponentRegistry


def _patched_registry(monkeypatch, reg: ComponentRegistry) -> None:
    import services.api_gateway.main as gateway_main
    monkeypatch.setattr(gateway_main, "registry", reg)


def _stub_realm_scoped_base(monkeypatch):
    """Replace the Realm/corpus_id-aware retriever/generator resolution with
    plain stubs — this file is about domain-pack hook wiring, not about
    which infra chat talks to (see test_realm_scoping.py's chat-specific
    coverage for that)."""
    import services.api_gateway.main as gateway_main
    from adapters.generator_stub import GeneratorStub
    from adapters.qdrant import QdrantRetrieverStub

    async def fake_retriever(realm_id, corpus_id, embedder, pipeline_id="naive"):
        return QdrantRetrieverStub()

    async def fake_generator(realm_id):
        return GeneratorStub()

    monkeypatch.setattr(gateway_main, "_build_realm_scoped_retriever", fake_retriever)
    monkeypatch.setattr(gateway_main, "_build_realm_scoped_generator", fake_generator)


async def test_grounder_is_wired_when_pack_has_a_refusal_policy(monkeypatch) -> None:
    import services.api_gateway.main as gateway_main
    from core.grounding import TokenOverlapGrounder

    reg = ComponentRegistry()
    reg.register("embedder", "bge_m3", object())
    reg.register("grounder", "token_overlap", TokenOverlapGrounder())
    reg.register("domain_hooks", "manuals", {"refusal_policy": ThresholdRefusalPolicy()})
    _patched_registry(monkeypatch, reg)
    _stub_realm_scoped_base(monkeypatch)

    pipeline = await gateway_main._build_chat_pipeline(["manuals"])

    assert pipeline._refusal_policy is not None
    assert pipeline._grounder is not None
    assert isinstance(pipeline._grounder, TokenOverlapGrounder)


async def test_no_grounder_resolve_attempt_when_pack_has_no_refusal_policy(monkeypatch) -> None:
    """Don't pay the grounding cost on every chat turn when nothing would
    consult its result anyway."""
    import services.api_gateway.main as gateway_main

    reg = ComponentRegistry()
    reg.register("embedder", "bge_m3", object())
    # Deliberately no "grounder" registered — if _build_chat_pipeline tried
    # to resolve one here it would raise KeyError; it shouldn't even try.
    reg.register("domain_hooks", "generic_qa", {"mask_engine": object()})
    _patched_registry(monkeypatch, reg)
    _stub_realm_scoped_base(monkeypatch)

    pipeline = await gateway_main._build_chat_pipeline(["generic_qa"])

    assert pipeline._refusal_policy is None
    assert pipeline._grounder is None


async def test_missing_grounder_degrades_gracefully_instead_of_raising(monkeypatch) -> None:
    """If a pack has a refusal_policy but the token_overlap grounder somehow
    isn't registered, _build_chat_pipeline must not 500 the request — same
    graceful-degradation contract as _maybe() elsewhere in this codebase."""
    import services.api_gateway.main as gateway_main

    reg = ComponentRegistry()
    reg.register("embedder", "bge_m3", object())
    reg.register("domain_hooks", "manuals", {"refusal_policy": ThresholdRefusalPolicy()})
    _patched_registry(monkeypatch, reg)
    _stub_realm_scoped_base(monkeypatch)

    pipeline = await gateway_main._build_chat_pipeline(["manuals"])  # must not raise

    assert pipeline._refusal_policy is not None
    assert pipeline._grounder is None


async def test_chat_does_not_fetch_retrieval_pins(monkeypatch) -> None:
    """Inverted from the test that used to live here,
    which asserted chat DID apply pins.

    Chat is the closest thing the platform has to a real request path, and
    fetching pins made a served answer depend on the platform's database
    being reachable at query time. the design notes forbids
    that. Worse, the lookup degraded to an empty list on failure, so the
    answer changed silently with nothing reporting it.

    The lookup is booby-trapped rather than merely unasserted: an assertion
    on the resulting pipeline would still pass if the fetch happened and its
    result were dropped, and the round-trip is itself the thing being
    removed."""
    import services.api_gateway.main as gateway_main

    reg = ComponentRegistry()
    reg.register("embedder", "bge_m3", object())
    _patched_registry(monkeypatch, reg)
    _stub_realm_scoped_base(monkeypatch)

    async def exploding_load_active_pins(realm_id, corpus_id):
        raise AssertionError("chat must not fetch retrieval pins")

    monkeypatch.setattr(gateway_main.experiments, "_load_active_pins", exploding_load_active_pins)

    pipeline = await gateway_main._build_chat_pipeline([], realm_id="demo", corpus_id="handbook")

    assert pipeline._retrieval_pins == []


async def test_chat_does_not_fetch_retrieval_pins_for_the_configurable_pipeline_either(monkeypatch) -> None:
    import services.api_gateway.main as gateway_main

    reg = ComponentRegistry()
    reg.register("embedder", "bge_m3", object())
    reg.register("domain_hooks", "generic_qa", {"mask_engine": object()})
    _patched_registry(monkeypatch, reg)
    _stub_realm_scoped_base(monkeypatch)

    async def exploding_load_active_pins(realm_id, corpus_id):
        raise AssertionError("chat must not fetch retrieval pins")

    monkeypatch.setattr(gateway_main.experiments, "_load_active_pins", exploding_load_active_pins)

    pipeline = await gateway_main._build_chat_pipeline(["generic_qa"], realm_id="demo", corpus_id="handbook")

    assert pipeline._retrieval_pins == []
