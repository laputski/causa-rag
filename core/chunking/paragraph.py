"""Paragraph-level chunking strategy.

Splits on double newlines (\\n\\n). Falls back to sentence split for oversized paragraphs.
Works well for documents whose sections are separated by blank lines.
"""
from __future__ import annotations

from core.chunking.sentence import SentenceChunkingStrategy
from core.models import Chunk, Document, _chunk_id


class ParagraphChunkingStrategy:
    """Chunks by paragraph boundaries (double newline), with sentence fallback."""

    strategy_id = "paragraph"

    def __init__(self, max_chars: int = 1024, min_chars: int = 32) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        self.max_chars = max_chars
        self.min_chars = min_chars
        self._sentence = SentenceChunkingStrategy(max_chars=max_chars, min_chars=min_chars)

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.content
        if not text:
            return []

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[Chunk] = []
        char_pos = 0

        for para in paragraphs:
            start = text.find(para, char_pos)
            if start == -1:
                start = char_pos
            end = start + len(para)
            char_pos = end

            if len(para) < self.min_chars:
                continue

            if len(para) <= self.max_chars:
                chunks.append(Chunk(
                    chunk_id=_chunk_id(doc.source, start, end, self.strategy_id),
                    doc_id=doc.doc_id,
                    text=para,
                    start_char=start,
                    end_char=end,
                    strategy_id=self.strategy_id,
                    metadata=dict(doc.metadata),
                ))
            else:
                # fallback: sentence-split the oversized paragraph
                sub_doc = Document(
                    doc_id=doc.doc_id,
                    source=doc.source,
                    content=para,
                    metadata=doc.metadata,
                )
                sub_chunks = self._sentence.chunk(sub_doc)
                # fix start_char offsets relative to original doc
                for sc in sub_chunks:
                    sc.start_char += start
                    sc.end_char += start
                    sc.strategy_id = self.strategy_id
                    sc.chunk_id = _chunk_id(doc.source, sc.start_char, sc.end_char, self.strategy_id)
                chunks.extend(sub_chunks)

        return chunks
