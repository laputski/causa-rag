from __future__ import annotations

from core.models import Chunk, Document, _chunk_id


class FixedChunkingStrategy:
    """Splits document text into fixed-size overlapping windows.

    Completely domain-neutral: operates on raw text only.
    """

    strategy_id = "fixed"

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be in [0, chunk_size)")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.content
        step = self.chunk_size - self.overlap
        chunks: list[Chunk] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(doc.source, start, end, self.strategy_id),
                    doc_id=doc.doc_id,
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                    strategy_id=self.strategy_id,
                    metadata=dict(doc.metadata),
                )
            )
            if end == len(text):
                break
            start += step
        return chunks
