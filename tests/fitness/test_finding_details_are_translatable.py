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

#: Values that are not this platform's to translate, and the reason each is.
#:
#: Four values used to reach a reader in the server's English whatever their
#: language, and all four are closed now: three named a thing from a closed set
#: and needed an identifier beside the sentence, and the fourth was two
#: sentences under one identifier and needed to say which. What is left here is
#: one value that nobody can translate, because nobody here wrote it.
NOT_OURS_TO_TRANSLATE: dict[str, str] = {
    "unverified_coverage.note": (
        "a library's own exception message, shown as it arrived; parenthesised on the way "
        "out because only this side knows whether there is one"
    ),
}

#: Values that arrive as their parts, so the interface writes the clause and
#: not only the frame around it, and what each becomes when it does.
ASSEMBLED_BY_THE_INTERFACE: dict[str, str] = {
    "metric_without_grounds.grounds": (
        "one clause per metric; a precondition is one of three and carries an identifier, "
        "and a metric's own name stays as it is because that is what a reader searches for"
    ),
    "aggregate_disagrees.named": (
        "one clause per metric, in two shapes: a number the questions never carried at all, "
        "and a number they carried and disagreed with"
    ),
    "segmentation_broke_its_promise.unmet": (
        "one clause per promise; a promise belongs to exactly one segmentation strategy, so "
        "the strategy's own name identifies the promise it broke"
    ),
}

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][\w]*)\s*\}\}")


def _parts() -> dict[str, dict[str, dict[str, set[str]]]]:
    """For each finding, the params that arrive as a list, and for each shape
    of item in that list, the fields it carries.

    Read off the source, and off the dictionaries the list is built from, so a
    field renamed in the detector and not in the clause is caught by the same
    pass that catches a missing clause. A list may be built inside the call or
    appended to before it, so both are followed; the empty string is the shape
    of an item that does not name one.
    """
    tree = ast.parse((ROOT / "core" / "eval" / "detectors.py").read_text(encoding="utf-8"))
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    found: dict[str, dict[str, dict[str, set[str]]]] = {}
    for function in functions:
        for node in ast.walk(function):
            if not (isinstance(node, ast.Call)
                    and getattr(node.func, "id", "") == "DiagnosticItem"):
                continue
            keywords = {k.arg: k.value for k in node.keywords}
            identifier = keywords["id"].value
            key = keywords["detail_key"].value if "detail_key" in keywords else identifier
            params = keywords["params"]
            for name, value in zip(params.keys, params.values, strict=True):
                items = _dictionaries_behind(value, function)
                if not items:
                    continue
                shapes: dict[str, set[str]] = {}
                for item in items:
                    fields = {field.value for field in item.keys}
                    shape = next((v.value for f, v in zip(item.keys, item.values, strict=True)
                                  if f.value == "shape"), "")
                    shapes[shape] = fields - {"shape"}
                found.setdefault(key, {})[name.value] = shapes
    return found


def _dictionaries_behind(value: ast.expr, function: ast.FunctionDef) -> list[ast.Dict]:
    """The dictionary literals one param's value is built from, if it is a list
    of them at all.

    A list written inside the call is read from the call. A list named there
    and built above is followed to where it was built, whether that was one
    expression or a series of appends: a detector with two shapes of clause
    has to write it the second way, and one with a single shape usually
    writes it the first.
    """
    if isinstance(value, ast.ListComp | ast.List):
        return [node for node in ast.walk(value) if isinstance(node, ast.Dict)]
    if not isinstance(value, ast.Name):
        return []
    appended: list[ast.Dict] = []
    for node in ast.walk(function):
        if (isinstance(node, ast.Assign)
                and any(getattr(target, "id", "") == value.id for target in node.targets)
                and isinstance(node.value, ast.ListComp | ast.List)):
            appended.extend(inner for inner in ast.walk(node.value)
                            if isinstance(inner, ast.Dict))
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "append"
                and getattr(node.func.value, "id", "") == value.id
                and node.args and isinstance(node.args[0], ast.Dict)):
            appended.append(node.args[0])
    return appended


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


def test_each_value_nobody_can_translate_is_still_one() -> None:
    """The listing is a remainder, so it has to shrink by somebody doing the
    work and never by an entry quietly outliving its value."""
    emitted = _emitted()
    for qualified, why in NOT_OURS_TO_TRANSLATE.items():
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
        written = clauses[key]
        for name, shapes in params.items():
            for shape, fields in shapes.items():
                sentence = written if shape == "" else written.get(shape)
                assert isinstance(sentence, str), (
                    f"{lang}.{key} has no sentence for the shape {shape!r} of {name}"
                )
                named = set(PLACEHOLDER.findall(sentence))
                assert named == fields, (
                    f"{lang}.{key}.{shape} writes {sorted(named)} and the detector sends "
                    f"{sorted(fields)} in {name}"
                )
        if isinstance(written, dict):
            spare = sorted(set(written) - {s for shapes in params.values() for s in shapes})
            assert spare == [], f"{lang}.{key} writes a sentence for shapes nobody sends: {spare}"


@pytest.mark.parametrize("lang", ["en", "ru"])
def test_every_code_a_finding_can_carry_has_the_readers_word_for_it(lang: str) -> None:
    """A clause's part may be one of the platform's own codes. Rendered as the
    code it is, a translated clause still hands the reader an English word, or
    worse an identifier with underscores in it.

    Every closed set a finding draws a code from is read from the module that
    declares it. Written for preconditions alone first, this passed while a
    promise had no word in one language: the clause was that word and nothing
    else, so the finding read `structure_aware` and said nothing at all.
    """
    bundle = json.loads(
        (ROOT / "ui" / "src" / "i18n" / "locales" / f"{lang}.json").read_text(encoding="utf-8"))
    words = bundle["runDiagnostics"]["findingWord"]
    for what, codes in _the_codes_a_finding_can_carry().items():
        assert codes, f"no {what} could be read, so this checks nothing"
        missing = sorted(codes - set(words))
        assert missing == [], f"{lang} has no word for the {what} {missing}"


def _the_codes_a_finding_can_carry() -> dict[str, set[str]]:
    """Every closed set a finding draws a code from, read where it is declared.

    Not one list here: a set maintained beside the check is a second place to
    forget, and forgetting it looks exactly like having nothing to add.
    """
    from core.chunking.post_conditions import PROMISES

    return {
        "precondition": _constructed_with("core/eval/metric_definitions.py", "Precondition"),
        "promise": set(PROMISES),
        "reason coverage was not checked": _named_argument(
            "services/api_gateway/routers/experiments.py", "UnknownRefResolver", "reason_id")
        | _default_of("core/eval/ref_resolution.py", "reason_id"),
    }


def _constructed_with(path: str, name: str) -> set[str]:
    """The first positional argument of every construction of `name`."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    return {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == name
        and node.args and isinstance(node.args[0], ast.Constant)
    }


def _named_argument(path: str, call: str, argument: str) -> set[str]:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    return {
        keyword.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == call
        for keyword in node.keywords
        if keyword.arg == argument and isinstance(keyword.value, ast.Constant)
    }


def _default_of(path: str, field: str) -> set[str]:
    """The default a dataclass field carries, which is the case nobody names."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    return {
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == field
        and isinstance(node.value, ast.Constant)
    }
