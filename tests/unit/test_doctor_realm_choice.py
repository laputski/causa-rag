"""Which realm the installation check reports on must not depend on luck.

`check_realms` took `realms[0]`, that is whichever the database happened to
return first. A second realm therefore changed the subject of the check
silently, and the output never said which realm it had looked at. The advice it
printed on failure was the literal command that fills the demo realm, whatever
realm had actually been checked.
"""
from __future__ import annotations

from tools.doctor import _seed_hint, _subject_realm

WORK = [{"id": "alpha"}, {"id": "demo"}]
GROUND = {"id": "proving-ground", "purpose": "proving_ground"}


def test_the_choice_does_not_depend_on_the_order_the_database_returned() -> None:
    assert _subject_realm(WORK) == _subject_realm(list(reversed(WORK)))


def test_a_realm_built_to_be_broken_is_not_the_subject_of_a_health_check() -> None:
    """The proving ground is deliberately faulty; reporting on it would answer
    the wrong question with a red light.

    Decided by the realm's own `purpose` and no longer by a list of identifiers
    kept in the checker. A list is a second place to remember, and a realm named
    anything else would have been checked as if it were healthy.
    """
    assert _subject_realm([{"id": "demo"}, GROUND]) == "demo"
    assert _subject_realm([GROUND, {"id": "demo"}]) == "demo"


def test_a_faulty_realm_under_another_name_is_still_recognised() -> None:
    """The bait for the previous test. Keyed on the mark and not on the name,
    so a second proving ground called anything at all is still passed over."""
    other = {"id": "aaa-experiments", "purpose": "proving_ground"}
    assert _subject_realm([other, {"id": "zzz-work"}]) == "zzz-work"


def test_the_faulty_realm_is_preferred_against_and_not_excluded() -> None:
    """On a machine holding nothing else, reporting on it beats reporting on
    nothing."""
    assert _subject_realm([GROUND]) == "proving-ground"


def test_the_advice_names_what_fills_the_realm_that_was_checked() -> None:
    assert _seed_hint("demo") == "make demo"
    assert _seed_hint("proving-ground") == "make proving-ground"
    hint = _seed_hint("customer-a")
    assert "make demo" not in hint, "advice for one realm printed for another"
    assert "customer-a" in hint


def test_the_advice_names_a_command_that_exists() -> None:
    """Advice pointing at a target nobody defined wastes the reader's time at
    the exact moment the check has already told them something is wrong.

    Read out of the Makefile, never compared against a copy of it kept here: a
    copy would keep agreeing with itself after the target was renamed."""
    import pathlib
    import re

    makefile = (pathlib.Path(__file__).parents[2] / "Makefile").read_text(encoding="utf-8")
    targets = set(re.findall(r"^([A-Za-z0-9_.-]+):", makefile, re.MULTILINE))
    for realm_id in ("demo", "proving-ground"):
        hint = _seed_hint(realm_id)
        assert hint.startswith("make "), f"{realm_id}: {hint!r} is not a make target"
        assert hint.removeprefix("make ") in targets, (
            f"{realm_id}: advice names {hint!r}, and the Makefile has no such target"
        )
