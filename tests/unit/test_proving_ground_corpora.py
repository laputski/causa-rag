"""The two halves the proving ground is measured against.

They are parallel on purpose: the same ten sections, the same numbers, the same
questions, written in two languages. Language is then the only difference
between them, and a language failure is measured as a comparison of the pair
instead of two unrelated observations.

Three properties came out of measuring, never out of deciding, and each is a
constraint the corpora are shaped by:

- headings are numbered flat. Nested numbering makes a *healthy* corpus report a
  duplicate number, because the number is read from the leaf of a structural
  path and "1" and "1.1" reduce to the same one under two different headings;
- numbers run without a gap, so a gap put there on purpose is the only one;
- every section carries enough text to clear the heading-only threshold and keep
  the average clear of the too-short one.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from core.chunking.structure_aware import StructureAwareChunkingStrategy
from core.eval.corpus_health import analyze
from core.models import Document
from tools.corpus_mutate import DEFECTS, Corpus, CorpusCannotCarryDefect, mutate, read_corpus

GROUND = Path(__file__).resolve().parents[2] / "corpus" / "proving-ground"
LANGUAGES = ("base-ru", "base-en")

PROVOKES = {
    "flatten_headings": "no_structure",
    "duplicate_documents": "duplicates",
    "drop_a_numbered_document": "missing_structural_numbers",
    "repeat_a_structural_number": "duplicate_structural_numbers",
    "shrink_to_fragments": "too_short",
    "add_a_second_language": "mixed_language",
    # None, and deliberately. This defect's whole effect is on the graph: a
    # keyword shared by every unit joins each to all the others, so the edges
    # a link step produces grow with the square of the corpus. Corpus health
    # reads documents and chunks and knows nothing of a graph, so the honest
    # expectation here is silence, and the pair that proves the defect lives
    # in tests/proving_ground against a loaded graph.
    "repeat_a_phrase_in_every_document": None,
    # None as well, and for the same kind of reason. This one adds a document
    # of ordinary words, so nothing about the corpus on disk is malformed:
    # every length, number and heading is in order, and what it does is to a
    # lexical index. The pair that proves it runs a hybrid retrieval.
    "stuff_a_document_with_the_corpus_own_words": None,
}


def _findings(corpus: Corpus, code: str) -> set[str]:
    chunker = StructureAwareChunkingStrategy()
    chunks = []
    for name, text in corpus.items():
        chunks.extend(chunker.chunk(Document(
            source=f"{code}/{name}", content=text,
            content_hash=hashlib.sha256(text.encode()).hexdigest(),
            metadata={"source_code": code, "article_no": name.split(".")[0]},
        )))
    return {item.id for item in analyze(chunks).items if item.id != "ok"}


@pytest.fixture(scope="module", params=LANGUAGES)
def corpus(request) -> tuple[str, Corpus]:
    code = request.param
    loaded = read_corpus(GROUND / code)
    assert loaded, f"{code} is missing from {GROUND}"
    return code, loaded


def test_the_base_corpus_is_healthy(corpus) -> None:
    """The control. Every mutation below is measured as a difference from this,
    so a finding here would make every one of them unattributable."""
    code, loaded = corpus
    assert _findings(loaded, code) == set()


def test_headings_are_numbered_flat(corpus) -> None:
    """Nested numbering would make the healthy corpus report a duplicate, and a
    corpus whose control is already red measures nothing."""
    _, loaded = corpus
    nested = [
        name for name, text in loaded.items()
        if re.search(r"^#{2,6}\s+\d+\.\d", text, re.M)
    ]
    assert nested == [], f"documents numbering a subsection: {nested}"
    unnumbered = [
        name for name, text in loaded.items()
        if not re.search(r"^#\s+\d+\s", text, re.M)
    ]
    assert unnumbered == [], f"documents whose top heading carries no number: {unnumbered}"


def test_the_numbering_runs_without_a_gap(corpus) -> None:
    _, loaded = corpus
    numbers = sorted(
        int(re.search(r"^#\s+(\d+)\s", text, re.M).group(1))
        for text in loaded.values()
    )
    assert numbers == list(range(1, len(numbers) + 1)), (
        f"the numbering has a gap of its own: {numbers}"
    )


def test_the_two_languages_are_parallel() -> None:
    """A language difference is the only difference. Anything else and the pair
    stops being a comparison."""
    ru = read_corpus(GROUND / "base-ru")
    en = read_corpus(GROUND / "base-en")
    assert set(ru) == set(en), "the two corpora hold different documents"
    for name in ru:
        ru_number = re.search(r"^#\s+(\d+)\s", ru[name], re.M).group(1)
        en_number = re.search(r"^#\s+(\d+)\s", en[name], re.M).group(1)
        assert ru_number == en_number, f"{name}: numbered {ru_number} and {en_number}"
        ru_headings = len(re.findall(r"^#{1,6}\s", ru[name], re.M))
        en_headings = len(re.findall(r"^#{1,6}\s", en[name], re.M))
        assert ru_headings == en_headings, f"{name}: {ru_headings} headings against {en_headings}"


def test_each_corpus_holds_one_language(corpus) -> None:
    """Each half is monolingual, so `add_a_second_language` has something to do
    and the control has nothing to report."""
    code, loaded = corpus
    assert "mixed_language" not in _findings(loaded, code)


@pytest.mark.parametrize("defect", DEFECTS, ids=lambda d: d.name)
def test_the_base_corpus_can_carry_every_defect(defect, corpus) -> None:
    """The point of shaping the corpus this way.

    Two defects operate on structural numbers, and a corpus numbering its files
    and not its headings refuses them: the demo corpus does, and this one
    was built numbered so the proving ground can stage them.
    """
    code, loaded = corpus
    try:
        broken = mutate(loaded, defect.name)
    except CorpusCannotCarryDefect as refusal:
        pytest.fail(f"the base corpus cannot carry {defect.name}: {refusal}")
    expected = PROVOKES[defect.name]
    fired = _findings(broken, code)
    if expected is None:
        assert fired == set(), (
            f"{defect.name} on {code} is declared invisible to corpus health and provoked "
            f"{sorted(fired)}, so either the declaration or the defect is wrong"
        )
        return
    assert expected in fired, (
        f"{defect.name} on {code}: expected {expected}, saw {sorted(fired) or 'nothing'}"
    )
    assert fired == {expected}, (
        f"{defect.name} on {code} also provoked {sorted(fired - {expected})}"
    )


# ── the golden set and the corpus it asks about ──────────────────────────────

def test_every_reference_in_the_golden_set_resolves_against_its_corpus(corpus) -> None:
    """A fixture and the dataset that queries it are one unit.

    Moving either alone leaves a suite that collects, runs, and measures
    nothing: every retrieval metric comes back zero, and zero is what a broken
    system and a broken pairing both look like. This platform has had exactly
    that, on a set that asked about a corpus nobody had seeded.
    """
    import json

    code, loaded = corpus
    chunker = StructureAwareChunkingStrategy()
    chunks = []
    for name, text in loaded.items():
        chunks.extend(chunker.chunk(Document(
            source=f"corpus/proving-ground/{code}/{name}", content=text,
            content_hash=hashlib.sha256(text.encode()).hexdigest(),
            metadata={"source_code": code, "article_no": name.split(".")[0]},
        )))
    from core.eval.ref_resolution import build_ref_index

    index = build_ref_index(chunks)
    path = GROUND.parents[1] / "eval" / "golden" / f"{code}.v1.fast.jsonl"
    assert path.exists(), f"the golden set for {code} is missing: {path}"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    dangling = sorted(
        ref for row in rows for ref in row.get("article_refs", []) if ref not in index
    )
    assert dangling == [], f"{code}: references no chunk carries: {dangling}"


def test_the_golden_set_has_something_for_the_refusal_metric_to_count(corpus) -> None:
    """A set where every question is answerable never exercises refusal, and a
    run over it reports a refusal score that measured nothing."""
    import json

    code, _ = corpus
    path = GROUND.parents[1] / "eval" / "golden" / f"{code}.v1.fast.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    out_of_scope = [r for r in rows if not r.get("article_refs")]
    assert out_of_scope, "no question is out of scope, so refusal is never measured"
    assert len(out_of_scope) < len(rows), "every question is out of scope"


def test_the_two_question_sets_are_parallel() -> None:
    """Same questions, same expected sections, different language."""
    import json

    def load(code: str) -> dict[str, dict]:
        path = GROUND.parents[1] / "eval" / "golden" / f"{code}.v1.fast.jsonl"
        return {
            row["id"]: row
            for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        }

    ru, en = load("base-ru"), load("base-en")
    assert set(ru) == set(en), "the two sets ask different questions"
    for qid in ru:
        ru_sections = [r.split("/")[-1] for r in ru[qid]["article_refs"]]
        en_sections = [r.split("/")[-1] for r in en[qid]["article_refs"]]
        assert ru_sections == en_sections, f"{qid}: expects sections {ru_sections} and {en_sections}"
        assert ru[qid]["question"] != en[qid]["question"], f"{qid}: the same wording in both sets"


def test_only_the_document_heading_carries_a_number_and_that_is_why_F29_is_unstaged() -> None:
    """The measurement behind a catalogue entry that has no proving-ground pair.

    F29, a citation naming a fragment other than the one the answer used,
    is read by `citation_number_coverage`, which looks for the structural
    number of the retrieved fragment occurring in the answer text. Here only
    the top-level heading of each document is numbered; every subsection is a
    bare title. Retrieval returns subsections, so the signal has no candidate
    number to check and returns None, and a live run against a server that
    moved every citation scored exactly like its control. Measured twice,
    then traced to this.

    So F29 stays unstaged, and the reason is the corpus and not the
    instrument. Numbering the subsections would make it stageable and would
    redden this test, which is why the number is written down.
    """
    label = re.compile(r"\[([^\]]+)\]")
    number = re.compile(r"\d+(?:[.\-]\d+)*")
    for language in LANGUAGES:
        documents = sorted((GROUND / language).glob("*.md"))
        numbered = total = 0
        for path in documents:
            document = Document(doc_id=path.stem, source=str(path),
                                content=path.read_text(encoding="utf-8"), metadata={})
            for chunk in StructureAwareChunkingStrategy().chunk(document):
                total += 1
                found = label.search(chunk.structural_path or "")
                if found and number.findall(found.group(1)):
                    numbered += 1
        assert numbered == len(documents), (
            f"{language}: {numbered} of {total} chunks carry a numbered label, one per "
            "document expected. If this grew, F29 may now be stageable"
        )
        assert total > numbered * 4, (
            f"{language}: {numbered} numbered of {total}, so a retrieved fragment is "
            "usually unnumbered, which is the premise above"
        )
