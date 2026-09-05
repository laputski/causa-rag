"""The structural strategy must actually be structural.

The incident log says this was fixed. It was fixed in one half. The log records
that `structure_aware` had been silently splitting every real ingest into flat
windows, and describes the remedy in two parts: a parser belonging to a domain
pack, selected by an explicit flag, and the chunker deriving a tree from
markdown headings on its own for a corpus that needs no pack. The first was
built and tested. The second was written in the past tense and never built, so
a corpus ingested without the flag kept getting flat windows under a strategy
whose name promises otherwise.

The demo realm ships that way and ships it today: `make demo` passes no parser,
so the handbook was indexed as twenty-two chunks all carrying the path `root`.
The test the log calls the lock exercises the pack's parser on a hand-written
string and never touches this path at all.

So these read real corpus files. A fixture proves the code does what the
fixture was shaped for; a corpus file proves it does what the product does.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from core.chunking.structure_aware import (
    StructureAwareChunkingStrategy,
    tree_from_markdown_headings,
)
from core.eval.corpus_health import analyze
from core.models import Document

CORPUS = Path(__file__).resolve().parents[2] / "corpus" / "demo_handbook"


def _document(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    return Document(
        source=str(path), content=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        metadata={"source_code": path.parent.name, "article_no": path.stem},
    )


def _body_characters(text: str) -> str:
    """Everything a reader would call the text: no headings, no whitespace."""
    return "".join(
        line for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ).replace(" ", "")


@pytest.fixture(scope="module")
def corpus_files() -> list[Path]:
    files = sorted(CORPUS.glob("*.md"))
    assert files, f"the demo corpus is missing from {CORPUS}"
    return files


def test_a_real_corpus_file_gets_a_real_structural_path(corpus_files: list[Path]) -> None:
    """The regression itself, in one assertion.

    Every chunk of every file used to carry `root`, which is what the chunker
    writes when it was given no tree and found none. Corpus health reports that
    as "no structural path was parsed for any chunk", and a reader had no way to
    know it meant the strategy had quietly become a different one.
    """
    chunker = StructureAwareChunkingStrategy()
    flat: list[str] = []
    for path in corpus_files:
        for chunk in chunker.chunk(_document(path)):
            if chunk.structural_path in ("", "root"):
                flat.append(f"{path.name}: {chunk.chunk_id}")
    assert flat == [], (
        f"{len(flat)} chunks of the demo corpus carry no structural path, so the "
        "structural strategy is producing what the fixed one would"
    )


def test_the_path_names_the_heading_the_text_sits_under(corpus_files: list[Path]) -> None:
    chunker = StructureAwareChunkingStrategy()
    chunks = chunker.chunk(_document(corpus_files[0]))
    titles = [c.structural_path for c in chunks]
    assert any("[" in t for t in titles), f"no chunk names a heading: {titles}"
    # The first file's own headings, read from the file itself and not
    # asserted from memory.
    headings = {
        line.lstrip("# ").strip()
        for line in corpus_files[0].read_text(encoding="utf-8").splitlines()
        if line.startswith("#")
    }
    named = {t.split("[", 1)[1].rstrip("]") for t in titles if "[" in t}
    assert named <= headings, f"paths naming something that is not a heading: {named - headings}"


def test_every_line_of_the_corpus_reaches_some_chunk(corpus_files: list[Path]) -> None:
    """A node carrying both a lead paragraph and subsections used to lose the
    paragraph: only a leaf was emitted.

    Measured before the fix: eight hundred characters of body text over eight
    files, one paragraph per section, absent from the index with nothing
    anywhere saying so. This is the class of failure the whole platform exists
    to catch, and it was in the platform.

    Checked line by line and not by a character count. A count was the first
    version of this test and it compared totals with a `>=`, which a loss of a
    hundred characters passes as soon as chunk overlap repeats two hundred. A
    total can be made right by the wrong text.
    """
    chunker = StructureAwareChunkingStrategy()
    for path in corpus_files:
        document = _document(path)
        produced = " ".join(c.text for c in chunker.chunk(document))
        missing = [
            line.strip() for line in document.content.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
            and line.strip() not in produced
        ]
        assert missing == [], (
            f"{path.name}: {len(missing)} lines of body text never reached a chunk, "
            f"beginning with {missing[0][:60]!r}"
        )


def test_a_section_with_subsections_still_yields_its_own_text() -> None:
    """The same defect, stated directly instead of by a character count."""
    tree = tree_from_markdown_headings(
        "# Purchase approval\nWho may approve, in one paragraph.\n"
        "## Thresholds\nBand C is approved by the finance director.\n"
    )
    assert tree is not None
    chunks = StructureAwareChunkingStrategy().chunk(
        Document(source="s", content="", content_hash="h", structure=tree)
    )
    texts = [c.text for c in chunks]
    assert any("one paragraph" in t for t in texts), (
        f"the section's own paragraph was dropped: {texts}"
    )
    assert any("finance director" in t for t in texts)


def test_text_before_the_first_heading_is_kept() -> None:
    """The pack's parser discards it, having no section to put it in. A
    document whose opening paragraph never reaches the index is the same silent
    loss by another route."""
    tree = tree_from_markdown_headings("An opening paragraph.\n\n# A section\nIts body.\n")
    assert tree is not None
    chunks = StructureAwareChunkingStrategy().chunk(
        Document(source="s", content="", content_hash="h", structure=tree)
    )
    assert any("opening paragraph" in c.text for c in chunks), (
        f"the preamble was dropped: {[c.text for c in chunks]}"
    )


def test_a_hash_inside_a_fenced_block_is_code_and_not_a_heading() -> None:
    tree = tree_from_markdown_headings(
        "# Real heading\nBody.\n\n```bash\n# not a heading, a shell comment\necho 1\n```\n"
    )
    assert tree is not None
    titles = [child.title for child in tree.children]
    assert titles == ["Real heading"], f"a comment inside a fence became a section: {titles}"


def test_a_document_with_no_headings_degrades_honestly() -> None:
    """Nothing is invented where there is nothing to read. The fallback is the
    flat window it always was, and corpus health says so."""
    assert tree_from_markdown_headings("Just a paragraph.\n\nAnd another.\n") is None
    chunks = StructureAwareChunkingStrategy().chunk(
        Document(source="s", content="Just a paragraph.\n\nAnd another.\n", content_hash="h")
    )
    assert [c.structural_path for c in chunks] == ["root"] * len(chunks)


# ── the bait ──────────────────────────────────────────────────────────────────

def test_corpus_health_reddens_on_the_flat_case_and_is_silent_on_the_derived_one(
    corpus_files: list[Path],
) -> None:
    """A guard nobody tried to fool is worth nothing.

    `health:no_structure` is what the catalogue names as the evidence for the
    failure "the chosen segmentation strategy silently does nothing" (F03). It
    has to speak when the strategy has quietly become the flat one, and stay
    quiet when it has not. Both halves are checked on the same real files.
    """
    chunker = StructureAwareChunkingStrategy()
    derived = [c for path in corpus_files for c in chunker.chunk(_document(path))]
    quiet = {item.id for item in analyze(derived).items}
    assert "no_structure" not in quiet, (
        f"the guard still reports a corpus that now has structure: {sorted(quiet)}"
    )

    # The same chunks with their paths flattened: nothing else differs.
    flattened = [c.model_copy(update={"structural_path": "root"}) for c in derived]
    loud = {item.id for item in analyze(flattened).items}
    assert "no_structure" in loud, (
        f"the guard is silent on a corpus with no structure at all: {sorted(loud)}"
    )


def test_two_sections_sharing_a_heading_do_not_share_an_identifier() -> None:
    """Deriving a tree made a collision reachable that had not been.

    A chunk is identified by its source, the strategy, and its offsets, and the
    offsets are counted inside the node. Two sections with the same heading in
    one file therefore agreed on everything the identifier was built from, and
    ingestion upserts by it, so one silently overwrote the other. Before the
    tree was derived, a document without one gave every chunk the path `root`
    and offsets running through the whole file, and no two could meet.

    This is the failure the catalogue lists as a lost source path making chunk
    identifiers collide. It was introduced by a fix and caught by auditing that
    fix and not by the suite, because a collision shows up as one chunk
    where two were expected and nothing anywhere reports the difference.
    """
    chunker = StructureAwareChunkingStrategy()
    document = Document(
        source="doc.md", content="# Notes\nFirst body.\n\n# Notes\nOther body.\n",
        content_hash="h",
    )
    chunks = chunker.chunk(document)
    assert len(chunks) == 2, f"expected two chunks, got {[c.text for c in chunks]}"
    assert len({c.chunk_id for c in chunks}) == 2, (
        "two sections sharing a heading produced one identifier, so ingestion "
        "would keep whichever was written last"
    )


def test_identifiers_stay_stable_when_the_document_does(corpus_files: list[Path]) -> None:
    """The property re-ingestion rests on: the same file gives the same
    identifiers, so loading it twice updates and does not add."""
    chunker = StructureAwareChunkingStrategy()
    for path in corpus_files[:3]:
        first = [c.chunk_id for c in chunker.chunk(_document(path))]
        second = [c.chunk_id for c in chunker.chunk(_document(path))]
        assert first == second, f"{path.name}: identifiers move between two runs on one file"


def test_no_two_chunks_of_the_corpus_share_an_identifier(corpus_files: list[Path]) -> None:
    chunker = StructureAwareChunkingStrategy()
    seen: dict[str, str] = {}
    for path in corpus_files:
        for chunk in chunker.chunk(_document(path)):
            clash = seen.get(chunk.chunk_id)
            assert clash is None, (
                f"{path.name} and {clash} produced the same identifier for different text"
            )
            seen[chunk.chunk_id] = path.name
