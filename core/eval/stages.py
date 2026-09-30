"""The stages one question's retrieval went through, and where the source it
needed was at each.

A stage is a ranked list of fragments that existed inside the pipeline while
it answered one question. In the order a fragment passes through them:

1. the halves a merge combines, each before the merge: `dense`, `sparse`, and
   `graph` for the graph pipeline. Parallel, not one after another, so a half
   that did not find a source did not lose it.
2. `fusion`, the window the merge produced, when there were halves to merge;
   `retrieval` otherwise, for a pipeline with one source. The window is the
   candidate list when one was recorded, else the list the reranker was given,
   else the final list.
3. `rerank`, the reranker's output, when a reranker ran.
4. `context`, what reached the model.

Only stages the run had are returned. Two consecutive stages holding the same
fragments in the same order are marked as the same, so a reader is not shown
one list twice as if it were two.

The rank of the source a question needs, on one stage, is the smallest
position among that stage's fragments whose reference id is one of the
question's references; None means none of them is there. With several
references the best of them counts, which is the question "did any source
this question needs get through".

Pure: no I/O, no model. A question stored before the halves were recorded
gets the stages from the window onwards and nothing is guessed about the
halves it did not record.
"""
from __future__ import annotations

from typing import Any

from core.eval.retrieval_metrics import extract_ref_id

#: The halves a merge can combine, in the order they are shown.
HALVES = ("dense", "sparse", "graph")


def _ids(refs: list[dict[str, Any]]) -> list[str]:
    return [str(ref.get("chunk_id") or "") for ref in refs]


def _rank(refs: list[dict[str, Any]], wanted: set[str]) -> int | None:
    for position, ref in enumerate(refs, start=1):
        if extract_ref_id(ref) in wanted:
            return position
    return None


def stages_of(question: dict[str, Any]) -> dict[str, Any]:
    """Every stage the question's retrieval had, its size, and the rank of
    the source the question needs on it, plus the stage where that source
    was lost.

    `lost_at` is the first stage after the halves where the source is absent,
    given that it was present on a half or an earlier stage. `found` says
    whether any stage had it at all. For a question about the corpus as a
    whole there is no bounded set of sources to rank, so ranks are None, and
    `lost_at` and `found` are None as well: nothing is claimed.
    """
    bounded = question.get("scope", "local") != "global"
    wanted = set(question.get("expected_refs") or []) if bounded else set()

    ordered: list[tuple[str, list[dict[str, Any]]]] = []
    for half in HALVES:
        refs = question.get(f"{half}_source_refs") or []
        if refs:
            ordered.append((half, refs))
    has_halves = bool(ordered)

    final = question.get("source_refs") or []
    pre_rerank = question.get("pre_rerank_source_refs") or []
    window = question.get("candidate_source_refs") or pre_rerank or final
    ordered.append(("fusion" if has_halves else "retrieval", window))
    if pre_rerank:
        ordered.append(("rerank", final))
    ordered.append(("context", final))

    stages: list[dict[str, Any]] = []
    for name, refs in ordered:
        previous = stages[-1] if stages and stages[-1]["stage"] not in HALVES else None
        same_as = (previous["stage"]
                   if previous is not None and previous["_ids"] == _ids(refs) else None)
        stages.append({
            "stage": name,
            "size": len(refs),
            "rank": _rank(refs, wanted) if wanted else None,
            "same_as": same_as,
            "_ids": _ids(refs),
        })
    for stage in stages:
        del stage["_ids"]

    if not wanted:
        return {"stages": stages, "lost_at": None, "found": None}

    seen = any(s["rank"] is not None for s in stages if s["stage"] in HALVES)
    lost_at = None
    for stage in (s for s in stages if s["stage"] not in HALVES):
        if stage["rank"] is not None:
            seen = True
        elif seen:
            lost_at = stage["stage"]
            break
    found = any(s["rank"] is not None for s in stages)
    return {"stages": stages, "lost_at": lost_at, "found": found}
