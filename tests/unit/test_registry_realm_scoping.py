"""The component registry, narrowed to one Realm.

There is one registry per process: the gateway serves every Realm at once, and
each Realm's active packs are loaded into it together. Activation, though, is
per Realm. Without narrowing, those two facts contradict each other on screen.
The new-run form on one Realm offered the internals of a pack a *different*
Realm had enabled, and nothing else, because those component kinds had exactly
one registered option each.
"""
from __future__ import annotations

import pytest

from core.domain import loader
from core.registry import ComponentRegistry


@pytest.fixture
def registry_with_two_packs():
    reg = ComponentRegistry()
    reg.register("route_policy", "naive", object())
    reg.register("reranker", "cross_encoder", object())
    loader.PACK_COMPONENTS.clear()
    # Two packs, so the test can show one realm's pack surviving while another
    # realm's disappears. Both ids are invented for this fixture: a real
    # installation's pack id names that installation's subject area, and those
    # belong in its own tree, and not in this repository's tests.
    loader.PACK_COMPONENTS["pack_a"] = {
        "route_policy": ["pack_a_question_type"], "scorer": ["pack_a_grounding"],
        "mask_engine": ["pack_a"], "refusal": ["pack_a"],
    }
    loader.PACK_COMPONENTS["pack_b"] = {"structure_parser": ["pack_b_section"]}
    for kinds in loader.PACK_COMPONENTS.values():
        for kind, ids in kinds.items():
            for i in ids:
                reg.register(kind, i, object())
    yield reg
    loader.PACK_COMPONENTS.clear()


def test_a_realm_sees_its_own_pack_and_the_platform_s_own_components(registry_with_two_packs) -> None:
    out = loader.components_outside_packs(registry_with_two_packs, ["pack_a"])
    assert out["route_policy"] == ["naive", "pack_a_question_type"]
    assert out["scorer"] == ["pack_a_grounding"]
    # The platform's own components always stay: they belong to no pack.
    assert out["reranker"] == ["cross_encoder"]
    # And another pack disappears entirely.
    assert out.get("structure_parser") == []


def test_a_realm_with_no_packs_sees_only_the_platform_s_own(registry_with_two_packs) -> None:
    out = loader.components_outside_packs(registry_with_two_packs, [])
    assert out["route_policy"] == ["naive"]
    assert out["scorer"] == []
    assert out["mask_engine"] == []
    assert out["reranker"] == ["cross_encoder"]


def test_a_pack_that_never_loaded_hides_nothing(registry_with_two_packs) -> None:
    # A pack absent from PACK_COMPONENTS was never loaded, so there is nothing
    # to subtract, and silently dropping components by name would be a guess.
    loader.PACK_COMPONENTS.pop("pack_b")
    out = loader.components_outside_packs(registry_with_two_packs, [])
    assert out["structure_parser"] == ["pack_b_section"]


def test_loading_a_pack_records_exactly_what_it_added() -> None:
    reg = ComponentRegistry()
    reg.register("route_policy", "naive", object())
    loader.PACK_COMPONENTS.clear()
    loader.load_active_packs(reg, {"active_packs": ["manuals"]})
    # Принадлежность считается разницей реестра до и после загрузки: сам
    # пакет о себе ничего не сообщает, а манифест перечисляет виды, не имена.
    assert "manuals" in loader.PACK_COMPONENTS
    assert "naive" not in loader.PACK_COMPONENTS["manuals"].get("route_policy", [])
    loader.PACK_COMPONENTS.clear()
