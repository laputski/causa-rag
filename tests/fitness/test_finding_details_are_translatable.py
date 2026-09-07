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
    "metric_without_grounds.told": (
        "one clause per metric, naming the metric, the questions and the precondition"
    ),
}

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][\w]*)\s*\}\}")


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
