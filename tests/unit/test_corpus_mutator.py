"""A mutation that provokes nothing is decoration.

The mutator exists so the two halves of a proving ground differ by one named
defect and by nothing else. That is a claim about behaviour, so it is tested as
one: every defect is applied to a real corpus, ingested through the real
chunker, and the corpus health checks are asked what they see.

Three of the six defects failed this test when it was first written, each for
its own reason, and the reasons are kept in the mutator's own comments. Two
changed the files and provoked no signal at all because they operated on
something the checks do not read. One provoked a different signal than it
claimed. A fourth put a second defect into the corpus by accident. None of
those would have been visible from reading the code.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from core.chunking.structure_aware import StructureAwareChunkingStrategy
from core.eval.corpus_health import analyze
from core.models import Document
from tools.corpus_mutate import (
    DEFECTS,
    Corpus,
    CorpusCannotCarryDefect,
    mutate,
    read_corpus,
    write_corpus,
)

CORPUS = Path(__file__).resolve().parents[2] / "corpus" / "demo_handbook"

# What each defect is meant to make the corpus health checks say. Kept beside
# the test and not inside the mutator, so the mutator cannot be edited into
# agreement with itself.
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


def _findings(corpus: Corpus) -> set[str]:
    chunker = StructureAwareChunkingStrategy()
    chunks = []
    for name, text in corpus.items():
        chunks.extend(chunker.chunk(Document(
            source=f"demo_handbook/{name}", content=text,
            content_hash=hashlib.sha256(text.encode()).hexdigest(),
            metadata={"source_code": "demo_handbook", "article_no": name.split(".")[0]},
        )))
    return {item.id for item in analyze(chunks).items if item.id != "ok"}


@pytest.fixture(scope="module")
def healthy() -> Corpus:
    corpus = read_corpus(CORPUS)
    assert corpus, f"the demo corpus is missing from {CORPUS}"
    return corpus


def test_the_healthy_corpus_provokes_nothing(healthy: Corpus) -> None:
    """The baseline every mutation is measured against. If this reports
    anything, every result below is attributable to the wrong thing."""
    assert _findings(healthy) == set(), (
        "the corpus the mutations start from is not healthy, so nothing below "
        "can be attributed to a mutation"
    )


@pytest.mark.parametrize("defect", DEFECTS, ids=lambda d: d.name)
def test_a_defect_provokes_the_signal_it_claims(defect, healthy: Corpus) -> None:
    """Half a bait. The other half is the test below."""
    try:
        broken = mutate(healthy, defect.name)
    except CorpusCannotCarryDefect:
        pytest.skip(f"{defect.name} needs a corpus that {defect.requires}")
    expected = PROVOKES[defect.name]
    fired = _findings(broken)
    if expected is None:
        assert fired == set(), (
            f"{defect.name} is declared invisible to corpus health and provoked "
            f"{sorted(fired)}, so either the declaration or the defect is wrong"
        )
        return
    assert expected in fired, (
        f"{defect.name} changed the corpus and the checks did not notice: "
        f"expected {expected}, saw {sorted(fired) or 'nothing'}"
    )


@pytest.mark.parametrize("defect", DEFECTS, ids=lambda d: d.name)
def test_a_defect_provokes_nothing_else(defect, healthy: Corpus) -> None:
    """One defect, and one only.

    A corpus carrying two defects makes a bait prove two things at once, which
    proves neither: a reader cannot tell which of them the signal answered. One
    mutation added its documents by repeating a single text, so the duplicate
    check fired beside the language one.
    """
    try:
        broken = mutate(healthy, defect.name)
    except CorpusCannotCarryDefect:
        pytest.skip(f"{defect.name} needs a corpus that {defect.requires}")
    extra = _findings(broken) - {PROVOKES[defect.name]}
    extra.discard(None)  # a defect declared invisible here subtracts nothing
    assert extra == set(), (
        f"{defect.name} also provoked {sorted(extra)}, so a bait on it would "
        "prove two things at once and neither in particular"
    )


def test_a_corpus_that_cannot_carry_a_defect_is_refused(healthy: Corpus) -> None:
    """Refusing beats returning a corpus that looks mutated and provokes nothing.

    Two defects repeat or drop a *structural* number, and the health check reads
    the number out of a chunk's structural path. This corpus numbers its files
    and not its headings, so both changed the files and were reported healthy.
    """
    with pytest.raises(CorpusCannotCarryDefect) as raised:
        mutate(healthy, "repeat_a_structural_number")
    assert "numbers its headings" in str(raised.value), (
        "the refusal does not say what shape of corpus would work"
    )


def test_an_unknown_defect_names_the_known_ones() -> None:
    with pytest.raises(KeyError) as raised:
        mutate({}, "make_it_worse")
    assert "flatten_headings" in str(raised.value)


def test_the_input_corpus_is_never_touched(healthy: Corpus) -> None:
    """The healthy half has to stay healthy: it is the control."""
    before = dict(healthy)
    for defect in DEFECTS:
        try:
            mutate(healthy, defect.name)
        except CorpusCannotCarryDefect:
            continue
    assert healthy == before


def test_a_written_corpus_reads_back_as_it_was_written(tmp_path: Path, healthy: Corpus) -> None:
    broken = mutate(healthy, "flatten_headings")
    destination = tmp_path / "broken"
    write_corpus(broken, destination)
    assert read_corpus(destination) == broken


def test_writing_over_a_directory_leaves_nothing_of_the_old_one(
    tmp_path: Path, healthy: Corpus,
) -> None:
    """A stale file surviving a rewrite is a corpus carrying two generations of
    a defect, and nothing would say so."""
    destination = tmp_path / "out"
    write_corpus(mutate(healthy, "duplicate_documents"), destination)
    write_corpus(mutate(healthy, "flatten_headings"), destination)
    assert read_corpus(destination) == mutate(healthy, "flatten_headings")


def test_every_defect_names_the_catalogue_entries_it_serves() -> None:
    """A mutation nobody has a use for is visible as such."""
    from core.eval.atlas import get

    for defect in DEFECTS:
        assert defect.provokes, f"{defect.name} serves no catalogue entry"
        for failure_id in defect.provokes:
            assert get(failure_id) is not None, (
                f"{defect.name} names {failure_id}, which is not in the catalogue"
            )
