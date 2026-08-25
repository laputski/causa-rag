from core.chunking.structure_aware import StructureAwareChunkingStrategy
from core.interfaces import ChunkingStrategy
from core.models import Document, DocumentNode


def _doc(content: str = "", structure: DocumentNode | None = None) -> Document:
    return Document(source="test.md", content=content, structure=structure)


def _node(node_id: str, node_type: str, content: str = "", title: str | None = None, children: list[DocumentNode] | None = None) -> DocumentNode:
    return DocumentNode(
        node_id=node_id,
        node_type=node_type,
        content=content,
        title=title,
        children=children or [],
    )


def test_satisfies_protocol():
    assert isinstance(StructureAwareChunkingStrategy(), ChunkingStrategy)


def test_no_structure_falls_back_to_content():
    s = StructureAwareChunkingStrategy()
    chunks = s.chunk(_doc(content="plain text without structure"))
    assert len(chunks) == 1
    assert chunks[0].text == "plain text without structure"


def test_empty_doc_no_structure():
    s = StructureAwareChunkingStrategy()
    assert s.chunk(_doc(content="")) == []


def test_leaf_nodes_become_chunks():
    root = _node("r", "root", children=[
        _node("c1", "section", content="the first section"),
        _node("c2", "section", content="the second section"),
    ])
    chunks = StructureAwareChunkingStrategy().chunk(_doc(structure=root))
    assert len(chunks) == 2
    texts = {c.text for c in chunks}
    assert "the first section" in texts
    assert "the second section" in texts


def test_structural_path_recorded():
    root = _node("r", "chapter", children=[
        _node("s", "section", content="body text", title="Introduction"),
    ])
    chunks = StructureAwareChunkingStrategy().chunk(_doc(structure=root))
    assert chunks[0].structural_path.startswith("chapter/section")


def test_nested_tree():
    root = _node("r", "doc", children=[
        _node("s1", "section", children=[
            _node("p1", "paragraph", content="paragraph one"),
            _node("p2", "paragraph", content="paragraph two"),
        ]),
    ])
    chunks = StructureAwareChunkingStrategy().chunk(_doc(structure=root))
    assert len(chunks) == 2


def test_overflow_splits_long_leaf():
    long_text = "word " * 300  # ~1800 chars
    root = _node("r", "root", children=[
        _node("leaf", "paragraph", content=long_text),
    ])
    chunks = StructureAwareChunkingStrategy(max_chunk_size=512, fallback_overlap=0).chunk(
        _doc(structure=root)
    )
    assert len(chunks) > 1
    assert all(len(c.text) <= 512 for c in chunks)


def test_overflow_never_cuts_inside_a_sentence_when_paragraphs_exist():
    """Real bug: with no paragraph/sentence awareness, the old raw
    character-count split landed mid-word/mid-clause on long leaf nodes
    (observed live in Neo4j chunk text — a node ending mid-word with the
    same fragment duplicated at the start of the next). Every produced
    chunk must now end on a real paragraph/sentence boundary."""
    items = [f"clause number {i}, with some text inside it;" for i in range(20)]
    long_text = "\n\n".join(items) + "\n\nA closing sentence at the end."
    root = _node("r", "root", children=[_node("leaf", "paragraph", content=long_text)])
    chunks = StructureAwareChunkingStrategy(max_chunk_size=200, fallback_overlap=20).chunk(
        _doc(structure=root)
    )
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    for c in chunks:
        stripped = c.text.strip()
        assert stripped.endswith(";") or stripped.endswith("."), (
            f"chunk does not end on a clause/sentence boundary: {stripped[-30:]!r}"
        )
    # No chunk should start mid-word — every chunk's first character starts
    # a real unit (a "clause..." item or the closing sentence), not a
    # fragment continuing from inside the previous chunk's last word.
    for c in chunks:
        first_word = c.text.strip().split()[0]
        assert first_word in ("clause", "A")


def test_overflow_does_not_split_off_a_bare_numeral_as_its_own_chunk():
    """Real bug found live via the Neo4j graph community viewer: hundreds of
    chunks across unrelated documents had text == "1." — nothing else. The
    sentence-boundary regex treats a clause numeral's trailing "." as a
    sentence end (it's followed by a capitalized word), splitting "1." off
    from the numbered item's actual content whenever a long numbered-list
    paragraph (no blank lines between items) overflows max_chunk_size."""
    long_text = "1. " + ("Body of the first clause with no full stops inside comma " * 6)
    root = _node("r", "root", children=[_node("leaf", "paragraph", content=long_text)])
    chunks = StructureAwareChunkingStrategy(max_chunk_size=120, fallback_overlap=10).chunk(
        _doc(structure=root)
    )
    assert len(chunks) > 1
    for c in chunks:
        assert c.text.strip() not in ("1.", "1"), f"bare numeral split into its own chunk: {c.text!r}"
    assert chunks[0].text.startswith("1. Body")


def test_compound_numeral_is_also_merged_into_its_item_text():
    long_text = "2.1. " + ("Text of a sub-clause with no inner full stops comma " * 6)
    root = _node("r", "root", children=[_node("leaf", "paragraph", content=long_text)])
    chunks = StructureAwareChunkingStrategy(max_chunk_size=120, fallback_overlap=10).chunk(
        _doc(structure=root)
    )
    for c in chunks:
        assert c.text.strip() not in ("2.1.", "2.1"), f"bare numeral split into its own chunk: {c.text!r}"
    assert chunks[0].text.startswith("2.1. Text")


def test_overflow_falls_back_to_character_split_only_when_no_boundaries_exist():
    """A pathological single "sentence" with neither blank lines nor
    terminal punctuation has no boundary to land on — the character-count
    split (scoped to just this case, not the whole node) is the only
    option left, and must still respect max_chunk_size."""
    long_text = "word " * 300  # no '.', '!', '?', or '\n\n' anywhere
    root = _node("r", "root", children=[_node("leaf", "paragraph", content=long_text)])
    chunks = StructureAwareChunkingStrategy(max_chunk_size=100, fallback_overlap=10).chunk(
        _doc(structure=root)
    )
    assert len(chunks) > 1
    assert all(len(c.text) <= 100 for c in chunks)


def test_overflow_character_fallback_never_cuts_inside_a_word():
    """Real bug found live via the chunk-coherence LLM-judge diagnostic: a
    single long comma-separated clause (no internal '.', '!', '?' — razdel
    treats it as one sentence) that exceeds max_chunk_size used to be cut
    by raw character index, regularly landing mid-word on both the end of
    one window and the overlap-shifted start of the next (found in production as a
    long word split as "re|habilitation" across two stored chunks). Every
    window boundary must now land on a space."""
    words = ["word" + str(i) for i in range(120)]
    long_sentence = "1." + " " + ", ".join(words) + "."
    root = _node("r", "root", children=[_node("leaf", "paragraph", content=long_sentence)])
    chunks = StructureAwareChunkingStrategy(max_chunk_size=120, fallback_overlap=20).chunk(
        _doc(structure=root)
    )
    assert len(chunks) > 1
    all_words = set(words)
    for c in chunks:
        for token in c.text.replace(",", " ").replace(".", " ").split():
            stripped = token.lstrip("1").lstrip(".") if token not in all_words else token
            assert stripped in all_words or stripped == "", f"fragment looks cut mid-word: {stripped!r} in chunk {c.text!r}"


def test_strategy_id():
    s = StructureAwareChunkingStrategy()
    chunks = s.chunk(_doc(content="text"))
    assert all(c.strategy_id == "structure_aware" for c in chunks)


def test_p1_no_domain_terms_in_code():
    """Sanity: the module source must not contain hard-coded domain terms."""
    import inspect

    import core.chunking.structure_aware as mod
    src = inspect.getsource(mod)
    # The Russian terms stay: a domain leak into this module would arrive in the
    # vocabulary of the corpora it was built against, so those are the strings
    # worth looking for.
    forbidden = ["статья", "пункт", "нпа", "закон", "decree", "article", "legal"]
    for term in forbidden:
        assert term.lower() not in src.lower(), f"domain neutrality violated: '{term}' found in structure_aware.py"
