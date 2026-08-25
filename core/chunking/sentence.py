"""Sentence-level chunking strategy.

Groups sentences into chunks up to max_chars. Uses razdel.sentenize() — a
statistical Russian sentence segmenter (Natasha project) — instead of a
hand-rolled regex boundary, which mis-split bare list/clause numerals
("1.", "2.1.") off as their own near-empty "sentence" (see
core/chunking/structure_aware.py for the production bug this caused there).
"""
from __future__ import annotations

from razdel import sentenize

from core.models import Chunk, Document, _chunk_id


class SentenceChunkingStrategy:
    """Chunks document by sentence boundaries, grouped to max_chars."""

    strategy_id = "sentence"

    def __init__(self, max_chars: int = 512, min_chars: int = 64) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        self.max_chars = max_chars
        self.min_chars = min_chars

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.content
        if not text:
            return []

        sentences = [s.text.strip() for s in sentenize(text) if s.text.strip()]

        chunks: list[Chunk] = []
        current: list[str] = []
        current_len = 0
        current_start = 0
        char_pos = 0

        for sent in sentences:
            sent_len = len(sent)
            if current and current_len + sent_len + 1 > self.max_chars:
                chunk_text = " ".join(current)
                end = current_start + len(chunk_text)
                if len(chunk_text) >= self.min_chars:
                    chunks.append(Chunk(
                        chunk_id=_chunk_id(doc.source, current_start, end, self.strategy_id),
                        doc_id=doc.doc_id,
                        text=chunk_text,
                        start_char=current_start,
                        end_char=end,
                        strategy_id=self.strategy_id,
                        metadata=dict(doc.metadata),
                    ))
                current = [sent]
                current_len = sent_len
                current_start = char_pos
            else:
                if not current:
                    current_start = char_pos
                current.append(sent)
                current_len += sent_len + 1
            char_pos += sent_len + 1

        if current:
            chunk_text = " ".join(current)
            end = current_start + len(chunk_text)
            if len(chunk_text) >= self.min_chars:
                chunks.append(Chunk(
                    chunk_id=_chunk_id(doc.source, current_start, end, self.strategy_id),
                    doc_id=doc.doc_id,
                    text=chunk_text,
                    start_char=current_start,
                    end_char=end,
                    strategy_id=self.strategy_id,
                    metadata=dict(doc.metadata),
                ))

        return chunks
