"""Structure-aware chunking strategy.

Traverses the DocumentNode tree and emits one chunk per leaf node.
If a leaf is too long it falls back to paragraph/sentence-boundary packing
(never a raw character cut, unless a single sentence alone still exceeds
the limit — see _split_text).
The `structural_path` field records the breadcrumb (e.g. "section/subsection/paragraph").

Completely domain-neutral: node types come from the Document tree itself,
and the paragraph/sentence boundary heuristics below are generic text
properties (blank lines, terminal punctuation) — this class never
hard-codes domain terms.
"""
from __future__ import annotations

import re

from razdel import sentenize

from core.models import Chunk, Document, DocumentNode, _chunk_id

# razdel.sentenize() is a statistical Russian sentence segmenter (Natasha
# project) — replaces a hand-rolled regex boundary that mis-split bare
# list/clause numerals ("1.", "2.1.") off as their own near-empty
# "sentence" whenever a long numbered-list paragraph (no blank lines
# between items) overflowed max_chunk_size, producing hundreds of useless
# duplicate "1." chunks in production. razdel already keeps a numeral
# marker attached to the text that follows it (verified directly), so the
# old _BARE_NUMERAL_RE merge step is no longer needed for that case — kept
# only as a defensive fallback in _sentence_units() for any edge case
# razdel itself doesn't cover (e.g. the one corpus source with Belarusian
# headers, which razdel has no official support for).
_BARE_NUMERAL_RE = re.compile(r"^\d+(?:[.\-]\d+)*\.?$")


# A markdown heading, and a fence that suspends them. A line starting with a
# hash inside a fenced block is code, not a heading, and reading it as one
# would invent a section out of a comment.
_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(\S.*)$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def tree_from_markdown_headings(content: str) -> DocumentNode | None:
    """A structural tree derived from markdown headings, or None when there are none.

    A heading convention belonging to a subject area belongs to a domain pack,
    and this is not one: `#` is a property of the file format, the same in every
    subject, so deriving from it needs no knowledge the platform does not have.

    Written because the incident log said it already existed. It did not. The
    log recorded that `structure_aware` had been silently splitting every real
    ingest into flat windows and described the fix in two halves, of which one
    was built: a parser was added to a domain pack and selected by an explicit
    flag. The other half, this one, was described in the past tense and never
    written, so a corpus ingested without that flag still got flat windows under
    a strategy whose whole name promises otherwise.

    Observed before writing this: the demo corpus has five headings in its first
    file and produced twenty-two chunks, every one of them carrying the path
    `root`.

    Text standing before the first heading is kept, in a node of its own. The
    pack's parser drops it, because a line arriving before any heading has no
    section to belong to and is discarded; here it becomes a preamble, since a
    document whose first paragraph never reaches the index is the same silent
    loss this module exists to stop.
    """
    root = DocumentNode(node_id="root", node_type="document", level=0)
    stack: list[DocumentNode] = [root]
    preamble: list[str] = []
    seen_heading = False
    in_fence = False

    for raw in content.splitlines():
        if _FENCE.match(raw):
            in_fence = not in_fence
        match = None if in_fence else _MARKDOWN_HEADING.match(raw)
        if match is None:
            if seen_heading:
                node = stack[-1]
                node.content = f"{node.content}\n{raw}".strip() if node.content else raw.strip()
            else:
                preamble.append(raw)
            continue

        seen_heading = True
        level = len(match.group(1))
        title = match.group(2).strip()
        while len(stack) > 1 and stack[-1].level >= level:
            stack.pop()
        node = DocumentNode(
            node_id=f"{len(stack[-1].children) + 1}",
            node_type="section",
            title=title,
            level=level,
        )
        stack[-1].children.append(node)
        stack.append(node)

    if not seen_heading:
        return None

    text = "\n".join(preamble).strip()
    if text:
        # First, so the document reads in its own order.
        root.children.insert(0, DocumentNode(
            node_id="0", node_type="preamble", level=1, content=text,
        ))
    return root


def _build_path(ancestors: list[DocumentNode], current: DocumentNode) -> str:
    parts = [n.node_type for n in ancestors] + [current.node_type]
    titles = [n.title for n in ancestors + [current] if n.title]
    if titles:
        return "/".join(parts) + f"[{titles[-1]}]"
    return "/".join(parts)


class StructureAwareChunkingStrategy:
    """Chunks a document by its structural tree (DocumentNode hierarchy).

    Falls back to fixed-size splitting for nodes whose text exceeds max_chunk_size.
    """

    strategy_id = "structure_aware"

    def __init__(self, max_chunk_size: int = 1024, fallback_overlap: int = 64) -> None:
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")
        self.max_chunk_size = max_chunk_size
        self.fallback_overlap = fallback_overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        structure = doc.structure
        if structure is None:
            # A caller that supplied no tree may still have given a document
            # that carries one in its own markup. Reading it costs one pass and
            # is the difference between this strategy doing what its name says
            # and doing exactly what `fixed` does under a different name.
            structure = tree_from_markdown_headings(doc.content)
        if structure is None:
            return self._split_text(
                doc.doc_id, doc.source, doc.content, structural_path="root", metadata=doc.metadata,
            )

        chunks: list[Chunk] = []
        self._traverse(doc.doc_id, doc.source, structure, ancestors=[], out=chunks, metadata=doc.metadata)
        return chunks

    def _traverse(
        self,
        doc_id: str,
        source: str,
        node: DocumentNode,
        ancestors: list[DocumentNode],
        out: list[Chunk],
        metadata: dict,
        ordinal: str = "0",
    ) -> None:
        path = _build_path(ancestors, node)
        # A node's own text is emitted whether or not it has children. Only a
        # leaf used to be emitted, so a section carrying an introduction before
        # its subsections lost that introduction: measured on the demo corpus,
        # eight hundred characters across eight files, one paragraph per
        # section, gone from the index without a word anywhere.
        #
        # It bites both ways of building a tree, this module's own and a domain
        # pack's, because both put a section's lead paragraph on the section and
        # its detail on the children.
        if node.content:
            out.extend(self._split_text(
                doc_id, source, node.content, structural_path=path, metadata=metadata,
                ordinal=ordinal,
            ))
        for index, child in enumerate(node.children):
            self._traverse(
                doc_id, source, child, ancestors + [node], out, metadata,
                ordinal=f"{ordinal}.{index}",
            )

    def _identity(self, structural_path: str, ordinal: str) -> str:
        """What makes one chunk's identifier different from another's.

        A chunk is identified by its source, this string, and its offsets, and
        the offsets are counted inside the node, never inside the document.
        The structural path alone is therefore not enough: two sections sharing
        a heading in one file produce the same path, both start at zero, and the
        two chunks collide. Ingestion upserts by that identifier, so one of them
        silently overwrites the other.

        Found by auditing this module's own change, which is what made the
        collision reachable: before a tree was derived, a document without one
        gave every chunk the path `root` and offsets running through the whole
        file, so no two could meet. The ordinal is the node's position in the
        tree, which distinguishes two sections that agree on everything a reader
        can see.
        """
        return f"{self.strategy_id}:{structural_path}:{ordinal}"

    def _split_text(
        self, doc_id: str, source: str, text: str, structural_path: str, metadata: dict,
        ordinal: str = "0",
    ) -> list[Chunk]:
        if not text:
            return []
        if len(text) <= self.max_chunk_size:
            return [
                Chunk(
                    chunk_id=_chunk_id(source, 0, len(text), self._identity(structural_path, ordinal)),
                    doc_id=doc_id,
                    text=text,
                    structural_path=structural_path,
                    strategy_id=self.strategy_id,
                    metadata=dict(metadata),
                )
            ]
        # Overflow: pack whole paragraphs/sentences into windows, never
        # cutting inside one — a raw character cut (the old behavior here)
        # routinely landed mid-word/mid-sentence on long leaf nodes (e.g. a
        # long enumerated list within one node), which is exactly
        # the kind of incoherent chunk a structure-aware strategy exists to
        # avoid. Falls all the way to a character cut only for the rare
        # single sentence that is itself still too long to fit a window.
        units = self._paragraph_units(text)
        chunks: list[Chunk] = []
        current: list[str] = []
        current_len = 0
        char_pos = 0
        current_start = 0

        def _flush() -> None:
            if not current:
                return
            chunk_text = "\n\n".join(current)
            end = current_start + len(chunk_text)
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(source, current_start, end, self._identity(structural_path, ordinal)),
                    doc_id=doc_id,
                    text=chunk_text,
                    structural_path=structural_path,
                    start_char=current_start,
                    end_char=end,
                    strategy_id=self.strategy_id,
                    metadata=dict(metadata),
                )
            )

        for unit in units:
            unit_start = text.find(unit, char_pos)
            if unit_start == -1:
                unit_start = char_pos
            char_pos = unit_start + len(unit)

            joiner_len = 2 if current else 0  # "\n\n" between packed units
            if current and current_len + joiner_len + len(unit) > self.max_chunk_size:
                _flush()
                current = [unit]
                current_len = len(unit)
                current_start = unit_start
            else:
                if not current:
                    current_start = unit_start
                current.append(unit)
                current_len += joiner_len + len(unit)
        _flush()
        return chunks

    def _paragraph_units(self, text: str) -> list[str]:
        """Splits into paragraphs (blank-line-separated), each further
        split into sentences if a single paragraph alone still exceeds
        max_chunk_size — these are the units _split_text packs into
        windows, so a window boundary always lands on a real paragraph or
        sentence boundary, never mid-word."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text]
        units: list[str] = []
        for para in paragraphs:
            if len(para) <= self.max_chunk_size:
                units.append(para)
            else:
                units.extend(self._sentence_units(para))
        return units

    def _sentence_units(self, text: str) -> list[str]:
        raw = [s.text.strip() for s in sentenize(text) if s.text.strip()]
        if not raw:
            raw = [text]
        # Merge a bare numeral with whatever follows it — it's a list/clause
        # marker, not a complete sentence (see _BARE_NUMERAL_RE comment).
        sentences: list[str] = []
        carry = ""
        for s in raw:
            if carry:
                sentences.append(f"{carry} {s}")
                carry = ""
            elif _BARE_NUMERAL_RE.match(s):
                carry = s
            else:
                sentences.append(s)
        if carry:
            sentences.append(carry)

        units: list[str] = []
        for sent in sentences:
            if len(sent) <= self.max_chunk_size:
                units.append(sent)
                continue
            # Last resort: a single sentence is itself too long (rare for
            # most prose, but common in long comma-separated enumerations
            # with only one terminal period). A character cut
            # is the only option left, but BOTH ends of each window are
            # snapped to the nearest space so neither the window's end nor
            # the next window's start (after applying fallback_overlap)
            # lands mid-word (found live: a Russian word split as "ре|абилитации"
            # across two chunks — the end was already snapped once, but the
            # overlap-shifted start of the next window still wasn't).
            # Falls back to the raw index only if no space exists in the
            # window at all (e.g. one pathologically long token) — without
            # that fallback, a search that never advances would loop forever.
            start = 0
            while start < len(sent):
                end = min(start + self.max_chunk_size, len(sent))
                if end < len(sent):
                    space = sent.rfind(" ", start, end)
                    if space > start:
                        end = space
                units.append(sent[start:end].strip())
                if end >= len(sent):
                    break
                next_start = max(end - self.fallback_overlap, start + 1)
                space = sent.find(" ", next_start, end)
                if space != -1:
                    next_start = space + 1
                start = next_start
        return units
