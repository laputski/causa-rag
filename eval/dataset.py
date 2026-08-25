"""EvalDataset — versioned question sets for RAG evaluation.

Format: JSONL, one question per line.
Each line: {"id": "...", "question": "...", "reference_answer": "...", "metadata": {...}}

Two speeds:
  - fast: 30–50 questions, runs in minutes, daily regression gate
  - full: 100+ questions, runs before release, includes span-level protocol
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalDataset:
    name: str
    version: str
    speed: str  # "fast" | "full"
    questions: list[dict[str, Any]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.questions)

    @classmethod
    def from_jsonl(cls, path: Path) -> EvalDataset:
        """Load from a JSONL file. Filename convention: {name}.{version}.{speed}.jsonl"""
        stem = path.stem  # e.g. "handbook.v1.fast"
        parts = stem.split(".")
        name = parts[0] if len(parts) > 0 else stem
        version = parts[1] if len(parts) > 1 else "v1"
        speed = parts[2] if len(parts) > 2 else "fast"

        questions: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                questions.append(json.loads(line))
        return cls(name=name, version=version, speed=speed, questions=questions)

    @classmethod
    def from_list(
        cls,
        questions: list[dict[str, Any]],
        name: str = "inline",
        version: str = "v0",
        speed: str = "fast",
    ) -> EvalDataset:
        return cls(name=name, version=version, speed=speed, questions=questions)

    def save_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for q in self.questions:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

    def subset(self, n: int) -> EvalDataset:
        """Return first n questions (for quick smoke tests)."""
        return EvalDataset(
            name=self.name,
            version=self.version,
            speed=self.speed,
            questions=self.questions[:n],
        )


def make_stub_dataset(n: int = 5, speed: str = "fast") -> EvalDataset:
    """Generate a minimal stub dataset for unit/integration tests."""
    questions = [
        {
            "id": f"q{i:03d}",
            "question": f"Test question number {i}?",
            "reference_answer": f"Test answer number {i}.",
            "metadata": {"type": "open"},
        }
        for i in range(n)
    ]
    return EvalDataset.from_list(questions, name="stub", version="v0", speed=speed)
