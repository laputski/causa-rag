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
        if doc.structure is None:
            return self._split_text(
                doc.doc_id, doc.source, doc.content, structural_path="root", metadata=doc.metadata,
            )

        chunks: list[Chunk] = []
        self._traverse(doc.doc_id, doc.source, doc.structure, ancestors=[], out=chunks, metadata=doc.metadata)
        return chunks

    def _traverse(
        self,
        doc_id: str,
        source: str,
        node: DocumentNode,
        ancestors: list[DocumentNode],
        out: list[Chunk],
        metadata: dict,
    ) -> None:
        path = _build_path(ancestors, node)
        if not node.children:
            out.extend(self._split_text(doc_id, source, node.content, structural_path=path, metadata=metadata))
        else:
            for child in node.children:
                self._traverse(doc_id, source, child, ancestors + [node], out, metadata)

    def _split_text(
        self, doc_id: str, source: str, text: str, structural_path: str, metadata: dict,
    ) -> list[Chunk]:
        if not text:
            return []
        if len(text) <= self.max_chunk_size:
            return [
                Chunk(
                    chunk_id=_chunk_id(source, 0, len(text), self.strategy_id + ":" + structural_path),
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
                    chunk_id=_chunk_id(source, current_start, end, self.strategy_id + ":" + structural_path),
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
