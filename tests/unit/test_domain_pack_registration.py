"""A pack really does get its components into the registry.

In an earlier version a pack's classes existed but resolved only in tests: neither the
gateway nor the runner knew about them. These close that gap and hold three
things in agreement: what the manifest declares, what `register()` registers,
and what can be pulled back out of the registry and under which kind.

The subject used to be a pack carrying one installation's own subject area,
which has moved to that installation's tree. It is now the technical-manuals pack, which ships with the platform
and can be switched on against the platform's own demo realm.
"""
from __future__ import annotations

import domain_packs.manuals as manuals
from core.answer.mask_engine import MaskEngine
from core.domain.loader import discover_packs
from core.registry import ComponentRegistry
from domain_packs.manuals.refusal import ManualsRefusalPolicy
from domain_packs.manuals.routing import ManualsRoutePolicy


def _fresh_registry() -> ComponentRegistry:
    reg = ComponentRegistry()
    manuals.register(reg, {})
    return reg


# @lat: [[domain-packs#Пример, совпадающий с демонстрацией]]
def test_every_kind_resolves_to_the_real_thing() -> None:
    reg = _fresh_registry()
    assert isinstance(reg.resolve("route_policy", "manual_question_type"), ManualsRoutePolicy)
    assert isinstance(reg.resolve("mask_engine", "manuals"), MaskEngine)
    assert isinstance(reg.resolve("refusal", "manuals"), ManualsRefusalPolicy)
    assert callable(reg.resolve("structure_parser", "manual_section"))


# @lat: [[domain-packs#Пример, совпадающий с демонстрацией]]
def test_domain_hooks_bundles_the_components_for_a_pack_agnostic_lookup() -> None:
    # Generic code (the /query handler, feedback triage) asks "what does this
    # pack contribute" by a single pack id and knows none of its components'
    # names.
    hooks = _fresh_registry().resolve("domain_hooks", "manuals")
    assert set(hooks) == {"route_policy", "mask_engine", "refusal_policy"}
    assert isinstance(hooks["refusal_policy"], ManualsRefusalPolicy)


# @lat: [[domain-packs#Манифест сверяется с тем, что пакет регистрирует]]
def test_the_manifest_says_what_the_pack_actually_registers() -> None:
    # The manifest was decorative: one pack registered an `error_taxonomy`
    # without declaring it and nothing stopped it. The list of kinds is a
    # promise to the Domain Packs screen, which displays it.
    from core.domain.loader import _BUILTIN_PACKS_DIR, load_pack

    # This repository's packs alone: an installation's own pack manifest is its
    # own business, and failing somebody else's build here would be a
    # surprise.
    for info in discover_packs(_BUILTIN_PACKS_DIR):
        reg = ComponentRegistry()
        load_pack(info.id, reg, {})
        actual = {k for k, v in reg.list_all().items() if v and k != "domain_hooks"}
        assert set(info.exported_kinds) == actual, (
            f"{info.id}: манифест {sorted(info.exported_kinds)}, "
            f"на деле {sorted(actual)}"
        )


# @lat: [[domain-packs#Пример, совпадающий с демонстрацией]]
def test_runner_resolves_a_pack_refusal_policy_under_the_kind_it_was_registered_as() -> None:
    """Regression: ExperimentRunner._build_pipeline has to resolve
    ExperimentConfig.refusal_policy under the same kind the pack registered it
    under ("refusal"). An earlier version called _maybe("refusal_policy"), a
    kind nothing registers under, and lost the component silently: the runner
    logged a warning, returned None, and did not fail. Only building the real
    pipeline catches this.
    """
    from core.experiment.config import ComponentRef, ExperimentConfig
    from core.experiment.runner import ExperimentRunner
    from core.pipeline import NaivePipeline

    reg = _fresh_registry()

    class _FakeComponent:
        def __init__(self, id_: str):
            self.embedder_id = self.retriever_id = self.generator_id = id_

        def embed(self, texts):
            return [[0.0] for _ in texts]

        def retrieve(self, query, k=5, filters=None, **kwargs):
            return []

        def generate(self, prompt):
            return "ok"

    fake = _FakeComponent("fake")
    reg.register("embedder", "fake", fake)
    reg.register("retriever", "fake", fake)
    reg.register("generator", "fake", fake)
    reg.register("pipeline", "naive", NaivePipeline(retriever=fake, embedder=fake, generator=fake))

    runner = ExperimentRunner(reg)
    cfg = ExperimentConfig(
        name="t",
        chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
        embedder=ComponentRef(kind="embedder", component_id="fake"),
        generator=ComponentRef(kind="generator", component_id="fake"),
        refusal_policy=ComponentRef(kind="refusal", component_id="manuals"),
    )
    pipeline = runner._build_pipeline(cfg)

    assert isinstance(pipeline._refusal_policy, ManualsRefusalPolicy)
