"""The service cards that make the proving ground's corpora compete.

The first fifteen documents of each corpus are general procedure. They answer a
question and do not make one hard: measured on them, the expected source ranked
first on every question. What follows is one card per instrument model, each
repeating the same six headings with that model's own numbers, which is what a
real handbook has beside its general procedure and what creates competition.

The cards are generated from a table and six templates. That makes one property
easy to lose, and it is asserted here instead of described: a template repeated
with different values produces identical text wherever a value was forgotten.
The first version opened every card with a preamble naming no model, and the
health check reported twenty-five duplicates on a corpus whose generator
described itself as carrying none.
"""
from __future__ import annotations

import collections
import re

import pytest

from tools.build_ground_corpus import MODELS, build


@pytest.fixture(scope="module")
def cards(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, str]]:
    """Written to a temporary place, so the test never depends on, nor
    disturbs, the corpus checked into the repository."""
    out = tmp_path_factory.mktemp("cards")
    build(destination=out)
    return {
        corpus_id: {p.name: p.read_text(encoding="utf-8")
                    for p in sorted((out / corpus_id).glob("*.md"))}
        for corpus_id in ("base-ru", "base-en")
    }


def _sections(text: str) -> list[str]:
    """The body under each heading, which is what becomes a chunk."""
    parts = re.split(r"^#{1,6} .*$", text, flags=re.M)
    return [p.strip() for p in parts if p.strip()]


@pytest.mark.parametrize("corpus_id", ["base-ru", "base-en"])
def test_no_two_sections_of_any_card_are_the_same_text(
    corpus_id: str, cards: dict[str, dict[str, str]]
) -> None:
    """The property the generator's docstring claims, asserted.

    Two identical sections are two duplicated chunks, which the health check
    reports and which would make every mutation of this corpus provoke a
    duplicate signal on top of the one it names.
    """
    bodies = [body for text in cards[corpus_id].values() for body in _sections(text)]
    repeated = {body[:60]: n for body, n in collections.Counter(bodies).items() if n > 1}
    assert repeated == {}, f"sections repeated across cards: {repeated}"


@pytest.mark.parametrize("corpus_id", ["base-ru", "base-en"])
def test_every_card_names_its_model_in_every_section(
    corpus_id: str, cards: dict[str, dict[str, str]]
) -> None:
    """The mechanism behind the property above, stated where it can be read.

    A section that never names the model is the one that will be forgotten and
    become a duplicate; asserting the mechanism says why, and asserting the
    property alone would only say that it broke.
    """
    for (name, text), model in zip(cards[corpus_id].items(), MODELS, strict=True):
        missing = [body[:50] for body in _sections(text) if model.code not in body]
        assert missing == [], f"{name}: sections naming no model: {missing}"


@pytest.mark.parametrize("corpus_id", ["base-ru", "base-en"])
def test_the_corpus_on_disk_is_numbered_from_one_without_a_gap(corpus_id: str) -> None:
    """The property that matters, stated over the corpus and not over the cards.

    A gap makes the structural check report a lost document and an overlap
    makes it report a repeated number. Both are defects this corpus exists to
    stage deliberately, so the healthy corpus must have neither.

    Written twice before it bit. The first version compared the card numbers
    against a range computed from the generator's own first-number constant, so
    moving the constant moved the expectation with it and a two-document gap
    passed. The second read the general documents as "everything below that
    constant", which had the same fault one level down. Nothing here reads the
    constant: the corpus on disk either runs from one without interruption or
    it does not.
    """
    import pathlib as _pathlib

    directory = _pathlib.Path(__file__).parents[2] / "corpus" / "proving-ground" / corpus_id
    numbers = sorted(int(p.stem) for p in directory.glob("*.md"))
    assert numbers, f"no documents in {directory}"
    assert numbers == list(range(1, len(numbers) + 1)), (
        f"the corpus is not a run from one: gaps or repeats at {sorted(set(range(1, numbers[-1] + 1)) - set(numbers))}"
    )


@pytest.mark.parametrize("corpus_id", ["base-ru", "base-en"])
def test_the_heading_number_matches_the_file_number(
    corpus_id: str, cards: dict[str, dict[str, str]]
) -> None:
    """The structural check reads the heading and the reference identity reads
    the file name. A disagreement between them makes a question's expected
    source unresolvable while everything still looks numbered."""
    for name, text in cards[corpus_id].items():
        first = text.splitlines()[0]
        assert first.startswith(f"# {name.split('.')[0]} "), f"{name} opens with {first!r}"


def test_the_two_languages_carry_the_same_models_in_the_same_order(
    cards: dict[str, dict[str, str]]
) -> None:
    """The pair exists so that language is the only difference between the two
    corpora. A model present in one and absent from the other would make every
    cross-language comparison measure two things at once."""
    for (ru_name, ru), (en_name, en), model in zip(
        cards["base-ru"].items(), cards["base-en"].items(), MODELS, strict=True
    ):
        assert ru_name == en_name
        assert model.code in ru and model.code in en


def test_every_model_code_is_used_once() -> None:
    """A repeated code would make two cards describe the same instrument with
    different numbers, which is a contradiction in the corpus and not a
    distractor."""
    codes = [m.code for m in MODELS]
    assert len(set(codes)) == len(codes)


def test_the_table_carries_enough_models_to_crowd_the_candidate_window() -> None:
    """Twenty-five cards of six sections put the corpus past two hundred
    chunks, so a candidate window of fifty stops covering it. Asserted because
    the number is the whole reason the table is this long."""
    assert len(MODELS) * 6 > 50 * 2
