"""The technical-manuals pack: the example that ships with the platform.

An example matching nothing anybody runs does not work as an example. The
platform's demo realm is a technical handbook, so this pack can be switched on
there and makes a visible difference.

The sample manual below is Russian on purpose: `_HEADING` accepts Cyrillic as
well as Latin section letters, and this is what exercises that branch.
"""
from __future__ import annotations

from core.domain.loader import load_pack
from core.registry import ComponentRegistry
from domain_packs.manuals.refusal import ManualsRefusalPolicy
from domain_packs.manuals.routing import classify_question_type
from domain_packs.manuals.structure_parser import parse_manual_section

MANUAL = """4 Обслуживание
Общие требования к обслуживанию.
4.2 Детектор
4.2.1 Замена детектора
Отключите питание перед снятием кожуха.
5 Техника безопасности
Не открывайте кожух под напряжением.
"""


def test_the_section_number_is_the_structural_path() -> None:
    tree = parse_manual_section(MANUAL)
    top = [c.node_id for c in tree.children]
    assert top == ["4", "5"]
    # Depth comes from the number itself: indentation does not survive
    # conversion out of a PDF, and a number does.
    assert [c.node_id for c in tree.children[0].children] == ["4.2"]
    assert [c.node_id for c in tree.children[0].children[0].children] == ["4.2.1"]


def test_text_under_a_heading_belongs_to_it() -> None:
    tree = parse_manual_section(MANUAL)
    assert "Отключите питание" in tree.children[0].children[0].children[0].content
    assert "Не открывайте кожух" in tree.children[1].content


def test_a_file_without_headings_yields_a_tree_rather_than_an_exception() -> None:
    # Ingestion has to survive a document that keeps no numbering: an exception
    # here would stop a whole corpus over one file.
    tree = parse_manual_section("Просто текст без нумерации.\nВторая строка.")
    assert tree.node_type == "document"
    assert tree.children == []


def test_a_lettered_appendix_is_a_heading_too() -> None:
    tree = parse_manual_section("A.1 Приложение\nСодержимое.")
    assert [c.node_id for c in tree.children] == ["A.1"]


def test_the_refusal_names_what_was_missing() -> None:
    from core.models import GroundingResult

    policy = ManualsRefusalPolicy()
    assert policy.should_refuse(GroundingResult(is_grounded=False, confidence=0.0))
    assert not policy.should_refuse(GroundingResult(is_grounded=True, confidence=0.9))
    text = policy.build_refusal("no_grounding").text
    # Somebody reading a manual came for an action: "rephrase your query" is no
    # use to them, and "check whether the manual is in the corpus" is an
    # instruction.
    assert "корпус" in text


def test_registering_the_pack_puts_all_four_kinds_in_the_registry() -> None:
    reg = ComponentRegistry()
    load_pack("manuals", reg, {})
    listed = reg.list_all()
    assert listed["structure_parser"] == ["manual_section"]
    assert listed["mask_engine"] == ["manuals"]
    assert listed["route_policy"] == ["manual_question_type"]
    assert listed["refusal"] == ["manuals"]
    # Generic code asks "what does this pack contribute" by pack id and knows
    # nothing about the names of its components.
    assert set(reg.resolve("domain_hooks", "manuals")) == {
        "route_policy", "mask_engine", "refusal_policy",
    }


def test_the_question_type_classifies_the_demo_realms_own_questions() -> None:
    """This pack is the example the demo realm switches on, so it has to work on
    that realm's questions. Its keyword lists were Russian only while the demo
    corpus is the English handbook, so all fifteen questions fell to `open`, the
    fallback, and switching the pack on changed nothing on screen.

    Asserting on more than one type, and not on a fixed mapping: which
    question is `closed` and which is `procedure` is a judgement the keyword
    list is allowed to revise, while everything landing in the fallback is the
    failure."""
    import json
    import pathlib

    bundle = json.loads((pathlib.Path(__file__).parents[2] / "ui" / "public" / "demo.realm.json")
                        .read_text(encoding="utf-8"))
    questions = [q["question"] for q in bundle["datasets"][0]["questions"]]
    types = {classify_question_type(q) for q in questions}
    assert len(types) > 1, f"every demo question classified as {types}"


def test_the_question_type_still_classifies_russian() -> None:
    """Adding English must not cost the language the pack was written for."""
    assert classify_question_type("Как заменить детектор?") == "procedure"
    assert classify_question_type("Есть ли ограничение по напряжению?") == "closed"
    assert classify_question_type("Опасно ли это?") == "safety"
