"""A finding's numbers travel beside its sentence, or the sentence stays English.

`title` and `action` were translated by identifier from the start. `detail` was
not, because it is composed on the server out of what the detector measured,
and a sentence with numbers in it cannot be looked up by identifier alone. So
it was the one line of a finding that a Russian screen showed in English, next
to two translated ones.

The values travel now, and this file is what keeps the two sides from drifting
apart in the three ways they can. A detector can send params for a sentence
nobody wrote. A sentence can name a placeholder no detector fills, which
renders as the placeholder's own name. And a detector can send a value the
sentence never uses, which is a number the reader was meant to see and does
not. All three are read off the source and off the bundles, never off a list
somebody maintains beside them.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.fitness

#: Values that are themselves a sentence the server composed, and the reason
#: each one is. They arrive inside a translated frame and stay English, which
#: is worth naming, and never worth leaving for a reader to notice: half a
#: line in one language is a smaller gap than a whole line, and still a gap.
#:
#: Closing one means giving the detector structured pieces instead of a joined
#: string, so the interface can compose that part too. Listed here so the work
#: is a known remainder and not a discovery.
COMPOSED_ON_THE_SERVER: dict[str, str] = {
    "unverified_coverage.reason": (
        "free text from whatever could not be checked, and not a sentence with a shape"
    ),
    "aggregate_disagrees.named": (
        "one clause per metric, each naming the metric and both numbers it disagreed between"
    ),
    "segmentation_broke_its_promise.unmet": (
        "one clause per promise the strategy declared and its own check did not find kept"
    ),
}

#: Values that arrive as their parts, so the interface writes the clause and
#: not only the frame around it, and what each becomes when it does.
#:
#: The fourth of the composed values above used to be one of them. Its clause
#: named a metric, a count and a precondition, and a precondition is one of
#: three, so an identifier beside its sentence was all that was missing.
ASSEMBLED_BY_THE_INTERFACE: dict[str, str] = {
    "metric_without_grounds.grounds": (
        "one clause per metric; a precondition is one of three and carries an identifier, "
        "and a metric's own name stays as it is because that is what a reader searches for"
    ),
}

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][\w]*)\s*\}\}")


def _parts() -> dict[str, dict[str, set[str]]]:
    """For each finding, the params that arrive as a list, and the fields of
    one item of that list.

    Read off the source, and off the dictionary the list is built from, so a
    field renamed in the detector and not in the clause is caught by the same
    pass that catches a missing clause.
    """
    tree = ast.parse((ROOT / "core" / "eval" / "detectors.py").read_text(encoding="utf-8"))
    found: dict[str, dict[str, set[str]]] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "DiagnosticItem"):
            continue
        keywords = {k.arg: k.value for k in node.keywords}
        identifier = keywords["id"].value
        key = keywords["detail_key"].value if "detail_key" in keywords else identifier
        for name, value in zip(keywords["params"].keys, keywords["params"].values, strict=True):
            if not isinstance(value, ast.ListComp | ast.List):
                continue
            item = next((inner for inner in ast.walk(value) if isinstance(inner, ast.Dict)), None)
            assert item is not None, (
                f"{key}.{name.value} is a list of something this cannot read, so the clause "
                "for it cannot be checked against what the detector sends"
            )
            found.setdefault(key, {})[name.value] = {field.value for field in item.keys}
    return found


def _emitted() -> dict[str, set[str]]:
    """Each detail key a detector can send, and the params it sends with it.

    Read out of the source with the parser and never by importing and calling,
    because a detector fires on a payload and half of these would need a
    payload built to provoke them.
    """
    tree = ast.parse((ROOT / "core" / "eval" / "detectors.py").read_text(encoding="utf-8"))
    found: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "DiagnosticItem"):
            continue
        keywords = {k.arg: k.value for k in node.keywords}
        assert "params" in keywords, (
            f"the finding at line {node.lineno} sends no params, so its detail cannot be "
            "composed anywhere but on the server"
        )
        identifier = keywords["id"].value
        key = keywords["detail_key"].value if "detail_key" in keywords else identifier
        found[key] = {name.value for name in keywords["params"].keys}
    return found


def _sentences(lang: str) -> dict[str, str]:
    bundle = json.loads(
        (ROOT / "ui" / "src" / "i18n" / "locales" / f"{lang}.json").read_text(encoding="utf-8"))
    flat: dict[str, str] = {}

    def walk(node: Any, prefix: str) -> None:
        for name, value in node.items():
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(value, dict):
                walk(value, path)
            else:
                flat[path] = value

    walk(bundle["runDiagnostics"]["findingDetail"], "")
    return flat


@pytest.mark.parametrize("lang", ["en", "ru"])
def test_every_finding_has_a_sentence(lang: str) -> None:
    missing = sorted(set(_emitted()) - set(_sentences(lang)))
    assert missing == [], f"{lang} has no sentence for {missing}, so they render the server's English"


@pytest.mark.parametrize("lang", ["en", "ru"])
def test_no_sentence_is_written_for_a_finding_nobody_sends(lang: str) -> None:
    orphaned = sorted(set(_sentences(lang)) - set(_emitted()))
    assert orphaned == [], f"{lang} keeps a sentence for a finding no detector sends: {orphaned}"


@pytest.mark.parametrize("lang", ["en", "ru"])
def test_every_placeholder_is_a_value_the_detector_sends(lang: str) -> None:
    """A placeholder nothing fills renders as its own name, in braces, to a
    reader who has no way to know that is what happened."""
    emitted = _emitted()
    for key, sentence in _sentences(lang).items():
        unfilled = sorted(set(PLACEHOLDER.findall(sentence)) - emitted[key])
        assert unfilled == [], f"{lang}.{key} names {unfilled}, which no detector fills"


@pytest.mark.parametrize("lang", ["en", "ru"])
def test_every_value_the_detector_sends_reaches_the_reader(lang: str) -> None:
    """The quiet half. A value measured, sent, and left out of the sentence is
    a number the English reader sees and the Russian one does not."""
    for key, params in _emitted().items():
        named = set(PLACEHOLDER.findall(_sentences(lang)[key]))
        dropped = sorted(params - named)
        assert dropped == [], f"{lang}.{key} never shows {dropped}, which the detector measured"


def test_each_value_composed_on_the_server_is_still_one() -> None:
    """The listing above is a remainder, so it has to shrink by somebody doing
    the work and never by the entry quietly outliving its value."""
    emitted = _emitted()
    for qualified, why in COMPOSED_ON_THE_SERVER.items():
        key, _, param = qualified.rpartition(".")
        assert key in emitted, f"{qualified} names a finding nothing sends"
        assert param in emitted[key], f"{qualified} names a value {key} no longer sends"
        assert len(why) > 40, f"{qualified} is listed with no reason worth reading"


def test_every_value_that_arrives_as_parts_is_declared_as_one() -> None:
    """A list nobody declared would be rendered by whatever the interface does
    with an unknown shape, which is `[object Object]` and not a sentence."""
    sent = {f"{key}.{name}" for key, params in _parts().items() for name in params}
    assert sent == set(ASSEMBLED_BY_THE_INTERFACE), (
        f"sent as parts: {sorted(sent)}; declared: {sorted(ASSEMBLED_BY_THE_INTERFACE)}"
    )
    for qualified, why in ASSEMBLED_BY_THE_INTERFACE.items():
        assert len(why) > 40, f"{qualified} is declared with no reason worth reading"


@pytest.mark.parametrize("lang", ["en", "ru"])
def test_every_value_that_arrives_as_parts_has_a_clause(lang: str) -> None:
    bundle = json.loads(
        (ROOT / "ui" / "src" / "i18n" / "locales" / f"{lang}.json").read_text(encoding="utf-8"))
    clauses = bundle["runDiagnostics"]["findingClause"]
    for key, params in _parts().items():
        assert key in clauses, f"{lang} has no clause for {key}, so its parts render as objects"
        named = set(PLACEHOLDER.findall(clauses[key]))
        for name, fields in params.items():
            assert named == fields, (
                f"{lang}.{key} writes {sorted(named)} and the detector sends {sorted(fields)} "
                f"in {name}"
            )


@pytest.mark.parametrize("lang", ["en", "ru"])
def test_a_field_naming_a_thing_has_the_readers_word_for_it(lang: str) -> None:
    """A clause's part may be one of the platform's own codes. Rendered as the
    code it is, a translated clause still hands the reader an English word."""
    from core.eval.metric_definitions import REACHED_THE_GENERATOR

    bundle = json.loads(
        (ROOT / "ui" / "src" / "i18n" / "locales" / f"{lang}.json").read_text(encoding="utf-8"))
    words = bundle["runDiagnostics"]["findingWord"]
    for name in _preconditions():
        assert name in words, f"{lang} has no word for the precondition {name!r}"
    assert REACHED_THE_GENERATOR.id in words, (
        "a precondition the catalogue's own detector names is missing a word"
    )


def _preconditions() -> set[str]:
    """Every precondition identifier, read off the module that declares them."""
    tree = ast.parse(
        (ROOT / "core" / "eval" / "metric_definitions.py").read_text(encoding="utf-8"))
    return {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "Precondition"
        and node.args and isinstance(node.args[0], ast.Constant)
    }
