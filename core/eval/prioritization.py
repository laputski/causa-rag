"""Turning a list of failures into an ordered list of work.

Root-cause analysis gave every failed question a cause. That is still a list of
complaints: thirty failures produce thirty verdicts, and an engineer reading
them has to notice for themselves that eleven of them name the same missing
document. Noticing that is the whole job here.

A task is one `(cause, entity)` pair. Its payoff is the number of questions
it would close, which is what makes tasks comparable to each other — the
thing a per-question verdict can never be.

The share of failures that reduce to a common cause at all is worth reporting
too. That number decides whether the last mile, exceptions and artefacts, is
worth starting: if almost every
failure is its own isolated case, generalisation has nothing to work with
and the effort belongs in data instead.

Pure: takes already-computed per-question verdicts, returns value objects.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# A ref id is one of `{source_code}/{article_no}`, `{source_code}#{path}`,
# `{doc_id}#{path}` or a bare `doc_id` (core/eval/retrieval_metrics.py
# #extract_ref_id). Everything before the first separator names the document;
# everything after names a place inside it.
_REF_SEPARATOR = re.compile(r"[/#]")


def entity_for_ref(ref: str) -> str:
    """The document a ref belongs to.

    Grouping by the whole ref id would put every question in its own group,
    since two questions rarely expect the exact same article — and a task
    list where every task closes one question is the list of complaints this
    module exists to replace. The document is the level at which work is
    actually done: a missing document is loaded once, a badly chunked one is
    re-ingested once, and both fix every question that referenced it.
    """
    return _REF_SEPARATOR.split(ref, maxsplit=1)[0].strip() or ref


def _refs_of(evidence: dict[str, Any]) -> list[str]:
    """Which refs a verdict was about.

    `refs` carries them since phase 2 made grouping possible. Runs stored
    before that have only the cause-specific lists — `absent_refs` for a
    missing document, `unknown_refs` for an unverifiable one — and those name
    refs just as well. Reading them keeps an older run's task list showing
    the document to fix instead of an anonymous group, which is most of what
    makes the list useful.
    """
    refs = evidence.get("refs")
    if refs:
        return [str(r) for r in refs]
    fallback: list[str] = []
    for key in ("absent_refs", "unknown_refs"):
        fallback.extend(str(r) for r in evidence.get(key) or [])
    return fallback


@dataclass(frozen=True)
class FixTask:
    """One piece of work, and what it would buy."""

    cause: str
    lever: str
    entity: str
    question_ids: tuple[str, ...]

    @property
    def questions(self) -> int:
        return len(self.question_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cause": self.cause,
            "lever": self.lever,
            "entity": self.entity,
            "questions": self.questions,
            "question_ids": list(self.question_ids),
        }


def group_failures(question_results: list[dict[str, Any]]) -> list[FixTask]:
    """Groups failed questions into tasks, most-reaching first.

    A question whose verdict names several documents joins the task of each
    of them, and is therefore counted more than once across tasks. That is
    deliberate: the question really would be closed by fixing any one of
    them, and making tasks add up to the failure count would understate the
    payoff of every task involved.

    Ties break on entity name so the order is stable between reads of the
    same run — an ordering that shuffles is one a reader cannot trust.
    """
    grouped: dict[tuple[str, str, str], list[str] ] = {}
    for qr in question_results:
        cause = qr.get("root_cause") or {}
        name = cause.get("cause")
        if not name:
            continue
        refs = _refs_of(cause.get("evidence") or {})
        qid = str(qr.get("question_id") or "")
        # A verdict with no refs (a question that had none at all) still
        # names a cause, and dropping it would hide failures from the very
        # list meant to account for all of them.
        entities = sorted({entity_for_ref(r) for r in refs}) or [""]
        for entity in entities:
            grouped.setdefault((name, str(cause.get("lever") or ""), entity), []).append(qid)

    tasks = [
        FixTask(cause=cause, lever=lever, entity=entity, question_ids=tuple(qids))
        for (cause, lever, entity), qids in grouped.items()
    ]
    return sorted(tasks, key=lambda t: (-t.questions, t.entity, t.cause))


def clusterable_share(tasks: list[FixTask]) -> float:
    """The share of failed questions that share a cause with at
    least one other question.

    Measured over distinct questions rather than over tasks, because a task
    count would reward splitting the same failures more finely. A question
    counts as clusterable when it belongs to any task closing more than one
    question, since that is exactly what makes generalisation possible.

    Returns 0.0 for no failures at all: nothing to cluster is not the same
    as nothing clusterable, but reporting a share of an empty set as
    anything other than zero would invite reading it as a verdict.
    """
    all_questions: set[str] = set()
    clustered: set[str] = set()
    for task in tasks:
        all_questions.update(task.question_ids)
        if task.questions > 1:
            clustered.update(task.question_ids)
    if not all_questions:
        return 0.0
    return len(clustered) / len(all_questions)
