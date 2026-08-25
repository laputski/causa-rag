"""The demo corpus and its golden dataset have to agree, or the demo teaches zeros.

`core/eval/retrieval_metrics.py#extract_ref_id` builds "{source_code}/{article_no}"
and drops any source_ref missing either half. `services/ingestion/cli.py` derives
`source_code` from the parent directory name and `article_no` from a numeric
filename stem. So a golden ref only ever resolves when a file with that exact
stem exists under `corpus/demo_handbook/`.

The first demo bundle shipped refs reading "раздел 2.1", which no ingest could
ever produce. Every retrieval metric read 0.0 and every question classified as
`uncovered`, which removes it from retrieval scoring entirely. A new user's
first run showed a column of zeros and nothing said why. These tests exist so
that cannot come back.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "corpus" / "demo_handbook"
GOLDEN_FILE = REPO_ROOT / "eval" / "golden" / "handbook.v1.fast.jsonl"

# services/ingestion/cli.py#_ARTICLE_NO_RE, reproduced so a change there that
# would silently stop deriving article_no fails here instead.
ARTICLE_NO_RE = re.compile(r"^\d+(\.\d+)*$")


@pytest.fixture(scope="module")
def questions() -> list[dict]:
    lines = GOLDEN_FILE.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@pytest.fixture(scope="module")
def corpus_files() -> list[Path]:
    return sorted(CORPUS_DIR.glob("*.md"))


# @lat: [[publication#The demo corpus is ingested by the CLI so that ref ids stay stable]]
def test_every_corpus_filename_yields_an_article_no(corpus_files):
    assert corpus_files, f"demo corpus is empty: {CORPUS_DIR}"
    for path in corpus_files:
        assert ARTICLE_NO_RE.match(path.stem), (
            f"{path.name} has a non-numeric stem, so ingestion would set "
            f"article_no=None and no golden ref could match it"
        )


def test_every_golden_ref_resolves_to_a_corpus_file(questions, corpus_files):
    available = {f"{CORPUS_DIR.name}/{p.stem}" for p in corpus_files}
    referenced = {ref for q in questions for ref in q["article_refs"]}
    unresolvable = referenced - available
    assert not unresolvable, (
        f"golden refs pointing at nothing: {sorted(unresolvable)}. "
        f"Available: {sorted(available)}"
    )


def test_every_corpus_document_is_exercised(questions, corpus_files):
    """A document nobody asks about is a document nobody notices breaking."""
    available = {f"{CORPUS_DIR.name}/{p.stem}" for p in corpus_files}
    referenced = {ref for q in questions for ref in q["article_refs"]}
    assert not (available - referenced), (
        f"documents with no question: {sorted(available - referenced)}"
    )


def test_dataset_carries_refusable_questions(questions):
    """`correct_refusal` is one of the demo realm's five key metrics.

    A question with no article_refs classifies as `out_of_scope`, which is what
    feeds that metric. Without any, the column sits empty on the demo's first
    run and the metric looks broken rather than unexercised.
    """
    out_of_scope = [q for q in questions if not q["article_refs"]]
    assert len(out_of_scope) >= 2, "expected at least two deliberately unanswerable questions"


def test_answerable_questions_carry_ground_truth(questions):
    answerable = [q for q in questions if q["article_refs"]]
    assert len(answerable) >= 10
    for q in answerable:
        assert q["ground_truth"].strip(), f"{q['id']} has no ground truth"
        assert q["question"].strip(), f"{q['id']} has no question"


def test_question_ids_are_unique(questions):
    ids = [q["id"] for q in questions]
    assert len(set(ids)) == len(ids), "duplicate question ids"


def test_demo_material_names_no_real_subject_area(questions, corpus_files):
    """The demo stays inside its own fictional company.

    A real subject area in the first thing anybody reads teaches that the
    platform is about that area, which is the opposite of what a demo is for.

    This checks for the marks of a subject area, and not for any particular
    installation's identifiers. Those belong to the publication guard, which
    scans this material along with everything else and stays in the private
    tree: a published test enumerating the names it forbids would hand a reader
    the list."""
    forbidden = ("statute", "ordinance", "plaintiff", "patient", "diagnosis",
                 "статья", "кодекс", "нпа", "пациент")
    for path in list(corpus_files) + [GOLDEN_FILE]:
        text = path.read_text(encoding="utf-8").lower()
        for word in forbidden:
            assert word not in text, f"{path.name} contains {word!r}"


def test_corpus_is_english(corpus_files):
    """The demo ships in the repository's primary language."""
    cyrillic = re.compile(r"[а-яА-ЯёЁ]")
    for path in corpus_files:
        assert not cyrillic.search(path.read_text(encoding="utf-8")), f"{path.name} contains Cyrillic"
