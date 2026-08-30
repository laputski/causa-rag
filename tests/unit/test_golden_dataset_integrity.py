"""Integrity invariants over eval/golden/*.jsonl (Eval Measurement
Trustworthiness, Phase 0). These are the control-question sets fed to
experiment runs — a duplicate id silently merges two different questions'
per-question metrics/regression, corrupting every run that uses the set.

The invariants are parametrised over whatever golden sets exist, so a new
set is covered the moment it is added. The one that motivated this file had
12 duplicate ids (same id, different question); that set has since been
removed along with the rest of the client corpora, and the check outlived it
because the failure mode belongs to the format rather than to one file.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

_GOLDEN_DIR = Path(__file__).parents[2] / "eval" / "golden"
_GOLDEN_FILES = sorted(_GOLDEN_DIR.glob("*.jsonl"))


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.parametrize("path", _GOLDEN_FILES, ids=lambda p: p.name)
def test_ids_are_unique(path: Path) -> None:
    rows = _load(path)
    ids = [r["id"] for r in rows]
    counts = Counter(ids)
    dupes = {k: v for k, v in counts.items() if v > 1}
    assert not dupes, f"{path.name} has duplicate ids: {dupes}"


@pytest.mark.parametrize("path", _GOLDEN_FILES, ids=lambda p: p.name)
def test_ground_truth_is_never_present_but_empty(path: Path) -> None:
    """An empty string is the dangerous state, not an absent key.

    answer_similarity scores whatever it is handed, so a question carrying
    "" as its reference answer reports a confident zero. A retrieval
    benchmark has no reference answers at all and says so by omitting the
    field, which the runner reads through a default and leaves unscored.
    """
    rows = _load(path)
    empty = [r["id"] for r in rows if "ground_truth" in r and not str(r["ground_truth"]).strip()]
    assert not empty, f"{path.name} has questions with an empty ground_truth: {empty}"


@pytest.mark.parametrize("path", _GOLDEN_FILES, ids=lambda p: p.name)
def test_a_set_answers_every_question_or_none_of_them(path: Path) -> None:
    """Half a set is worse than either whole.

    The runner averages each metric over the questions that produced it, so
    a set where only some questions carry a reference answer reports an
    answer_similarity computed over an arbitrary subset and labels it with
    the whole set's name. Either the set is scored for generation or it is
    a retrieval set; there is no readable middle.
    """
    rows = _load(path)
    answered = [r["id"] for r in rows if str(r.get("ground_truth") or "").strip()]
    assert len(answered) in (0, len(rows)), (
        f"{path.name} answers {len(answered)} of {len(rows)} questions; "
        "a partly-answered set reports generation metrics over an arbitrary subset"
    )


@pytest.mark.parametrize("path", _GOLDEN_FILES, ids=lambda p: p.name)
def test_article_refs_is_always_a_list(path: Path) -> None:
    """An absent list and an empty one mean different things.

    core/eval/answerability.py reads no refs as "out_of_scope", which feeds
    correct_refusal. A missing key would raise instead of classifying.
    """
    rows = _load(path)
    wrong = [r["id"] for r in rows if not isinstance(r.get("article_refs"), list)]
    assert not wrong, f"{path.name} has questions without an article_refs list: {wrong}"
