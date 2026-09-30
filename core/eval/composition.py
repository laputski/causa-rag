"""What a question set is made of.

The catalogue has an entry for a golden set too small and too uniform to
expose a defect, and something measures the first half of that: a comparison
says when a difference is smaller than the set's scatter. Nothing measured the
second. A set of nineteen questions about single facts, each answered by one
source, says nothing about a system built to answer questions that span the
corpus, however many of them it has.

This counts what a set holds, so the uniformity is visible before a run is
spent on it: questions by scope and by type, by answerability, how many
distinct sources they reference, how many carry assertions.

Distinct references and not a share of the corpus: a reference names a
document in one corpus and a numbered article in another, so dividing by the
corpus's document count would give a number whose meaning changes with the
corpus. Pure: no I/O.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from core.eval.answerability import resolve_answerability


def profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_scope: Counter[str] = Counter()
    by_type: Counter[str] = Counter()
    by_answerability: Counter[str] = Counter()
    references: set[str] = set()
    with_assertions = 0
    assertions = 0
    for row in rows:
        by_scope[row.get("scope") or "local"] += 1
        by_type[row.get("question_type") or "unlabelled"] += 1
        # Without an index to consult: the row's own declaration, else what
        # its references imply. A profile is read before any run.
        by_answerability[resolve_answerability(row, None)] += 1
        references.update(row.get("article_refs") or [])
        stated = [a for a in row.get("assertions") or [] if isinstance(a, str) and a.strip()]
        if stated:
            with_assertions += 1
            assertions += len(stated)
    return {
        "questions": len(rows),
        "by_scope": dict(by_scope),
        "by_type": dict(by_type),
        "by_answerability": dict(by_answerability),
        "distinct_references": len(references),
        "with_assertions": with_assertions,
        "assertions": assertions,
    }
