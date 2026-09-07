"""What each corpus defect is meant to make the corpus health checks say.

Kept out of `tools/corpus_mutate.py` on purpose: a mutator holding its own
expectations can be edited into agreement with itself, and then the pair proves
that the code matches the code. It is kept out of the two test modules that
read it for the reason the second one taught. The table stood in both, a new
defect was added to one of them, and the other stopped at a `KeyError` — a
second place to forget, which is the thing this repository catalogues.

`None` is a real answer and not a gap: some defects do nothing a check of the
documents can see, and their pair lives on the proving ground against a loaded
index or graph. Which of the two a defect is has to be written down, because
"provoked nothing" and "provoked nothing it was meant to" read alike.
"""
from __future__ import annotations

PROVOKES: dict[str, str | None] = {
    "flatten_headings": "no_structure",
    "duplicate_documents": "duplicates",
    "drop_a_numbered_document": "missing_structural_numbers",
    "repeat_a_structural_number": "duplicate_structural_numbers",
    "shrink_to_fragments": "too_short",
    "add_a_second_language": "mixed_language",
    "leave_every_other_section_a_heading": "header_only",
    "empty_the_corpus": "empty_corpus",
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
    # None, and this one is the reverse kind of pair. The corpus is well formed
    # by every measure these checks have: the lengths, the numbering and the
    # headings are all in order, and what is wrong is inside a fragment, where
    # a table has lost the row naming its columns. Nothing inspects a fragment
    # for that, which is what the catalogue entry says, so silence here is the
    # claim under test and not an omission. A separate test shows the table was
    # really cut, so the silence is about a corpus that carries the defect.
    "cut_a_table_and_a_list": None,
    # None, and for the reason its sibling above has: the whole of this one's
    # effect is on the graph, where a word shared by fifty units joins all
    # fifty. Nothing about the documents is malformed, and the checks here
    # read documents.
    "share_a_word_at_the_cap": None,
    # None. A very long document is not a malformed one, and at any chunk size
    # this platform uses by default it becomes many units of ordinary length.
    # The failure appears only when a load is given a chunk size above the
    # embedding model's window, so the pair for it lives with the loads.
    "pad_a_section_past_the_model_window": None,
    # None. Every length, number and heading stays in order and no text is
    # lost: the corpus reads as a more finely divided version of itself, which
    # is what makes the failure quiet. What it costs is measured on the index,
    # by asking how much of an answering section a window still holds.
    "split_every_section_in_two": None,
    # None. One sentence added to one document leaves every length, number and
    # heading where it was. What the sentence stages is measured on the index
    # and depends on the corpus: on a small one of short sections, nothing.
    "hide_a_code_in_one_document": None,
}
