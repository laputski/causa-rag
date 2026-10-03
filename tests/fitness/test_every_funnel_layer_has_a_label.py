"""Every layer the funnel can name has a label on both pages that show it.

The verdict is computed on the server and drawn by the interface, from a map
of layer to label kept on each page. A layer added on one side and not the
other is drawn as nothing: the question shows no badge, and a reader takes
that for a question with no verdict. A layer for a source the merge lost was
added, and nothing would have said the maps had fallen behind.
"""
from __future__ import annotations

import json
import re
import typing
from pathlib import Path

import pytest

from core.eval.funnel import Layer

ROOT = Path(__file__).resolve().parents[2]
PAGES = {
    "RunPage.tsx": "FUNNEL_BADGE",
    "ComparisonPage.tsx": "FUNNEL_LAYER_LABEL",
}


def labels_in(source: str, constant: str) -> dict[str, str]:
    """The layers a page's map names, each with the locale key it points at."""
    block = re.search(rf"const {constant}\b[^=]*=\s*\{{(.*?)\n\}}", source, re.S)
    assert block, f"no {constant} to read"
    return dict(re.findall(r"^\s*([a-z_]+):\s*(?:\{\s*label:\s*)?'([\w.]+)'", block.group(1), re.M))


def _lookup(tree: dict, dotted: str) -> object:
    for part in dotted.split("."):
        tree = tree.get(part) if isinstance(tree, dict) else None
    return tree


@pytest.mark.parametrize("page,constant", sorted(PAGES.items()))
def test_every_layer_is_drawn_and_named_in_both_languages(page: str, constant: str) -> None:
    labels = labels_in((ROOT / "ui" / "src" / "pages" / page).read_text(encoding="utf-8"), constant)
    layers = set(typing.get_args(Layer))
    assert layers - set(labels) == set(), f"{page} draws nothing for {sorted(layers - set(labels))}"
    for locale in ("en", "ru"):
        tree = json.loads((ROOT / "ui" / "src" / "i18n" / "locales" / f"{locale}.json").read_text(encoding="utf-8"))
        unnamed = sorted(layer for layer in layers if not _lookup(tree, labels[layer]))
        assert unnamed == [], f"{locale}: no words for {unnamed}"


def test_the_reader_of_a_page_s_map_finds_what_is_there() -> None:
    """The check reads TypeScript with a pattern, and a pattern that matched
    nothing would pass every page."""
    source = "const FUNNEL_BADGE: Record<string, X> = {\n  ok: { label: 'a.b', cls: 'c' },\n  rerank: 'd.e',\n}\n"
    assert labels_in(source, "FUNNEL_BADGE") == {"ok": "a.b", "rerank": "d.e"}
