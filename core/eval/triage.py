"""Feedback triage — turning a reviewer's free-text feedback comment into a
structured, cross-checked diagnosis. A reviewer's comment
("the dates are wrong and the citation points at the wrong document")
almost always bundles several distinct complaints together, each pointing
at a different pipeline layer and calling for a different fix; before any
fix is proposed, the comment needs to be pulled apart into classified,
individually checkable claims.

This module is pure logic only, the same boundary ``core/pins/overlay.py``
already draws: it builds the prompt an LLM should answer and parses/
validates whatever text comes back, and it performs the deterministic
cross-check against a run's own retrieval data — but it never calls a
generator itself (``services/api_gateway/routers/feedback.py`` owns that
one I/O call) and it never imports an error taxonomy directly (the caller
passes one in as plain data, so a different domain pack's own taxonomy
works with these exact same functions without this module knowing it
exists).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

# Below this confidence, a triage result is routed to manual review even
# when the model did return at least one recognized error class — a
# low-confidence guess is not the same as "no opinion" (needs_manual_review
# is still set), but it also is not withheld from the reviewer entirely.
_LOW_CONFIDENCE_THRESHOLD = 0.4


@dataclass
class TriageResult:
    """The structured outcome of triaging one feedback comment. The first
    five fields come directly from the model's own JSON response (see
    ``parse_triage_response``); ``verified_refs``/``funnel_layer``/
    ``diagnosis_confirmed`` are filled in afterward by ``cross_check``,
    which runs with no LLM involvement at all."""

    error_classes: list[str] = field(default_factory=list)
    target_spans: list[str] = field(default_factory=list)
    expected_refs: list[str] = field(default_factory=list)
    severity: str = "unknown"
    confidence: float = 0.0
    verified_refs: dict[str, bool] = field(default_factory=dict)
    funnel_layer: str | None = None
    diagnosis_confirmed: bool = False
    # Set whenever the model's response could not be trusted outright —
    # invalid/absent JSON, no recognized error class, or low confidence.
    # The caller's job is to route a `needs_manual_review=True` result to a
    # human rather than either hiding it or silently acting on it.
    needs_manual_review: bool = False
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_classes": self.error_classes,
            "target_spans": self.target_spans,
            "expected_refs": self.expected_refs,
            "severity": self.severity,
            "confidence": self.confidence,
            "verified_refs": self.verified_refs,
            "funnel_layer": self.funnel_layer,
            "diagnosis_confirmed": self.diagnosis_confirmed,
            "needs_manual_review": self.needs_manual_review,
            "parse_error": self.parse_error,
        }


def build_triage_prompt(comment: str, taxonomy: list[dict[str, Any]]) -> str:
    """Builds the prompt asking the Realm's own local generator to
    classify one feedback comment. ``taxonomy`` is a plain list of dicts,
    each with at least ``id``/``definition`` keys, supplied by the caller
    from whichever domain-pack-owned reference table applies — see this
    module's own docstring for why it is never imported here directly."""
    classes_desc = "\n".join(f"- {c['id']}: {c['definition']}" for c in taxonomy)
    return (
        "You are diagnosing a reviewer's comment about the quality of a RAG "
        "system's answer. Break the comment below into structured error "
        "classes. The comment may be written in any language; answer in the "
        "schema regardless.\n\n"
        f"Available error classes:\n{classes_desc}\n\n"
        f"Reviewer's comment: {comment!r}\n\n"
        "Reply STRICTLY as JSON with no text before or after it, using this "
        "schema:\n"
        '{"error_classes": ["<id of a class from the list above>", ...], '
        '"target_spans": ["<verbatim quote from the answer the complaint refers to>", ...], '
        '"expected_refs": ["<any reference, article or document the reviewer named>", ...], '
        '"severity": "low"|"medium"|"high", '
        '"confidence": <number from 0 to 1, how certain you are of this classification>}'
    )


def parse_triage_response(raw_text: str, valid_classes: frozenset[str] | set[str]) -> TriageResult:
    """Parses the model's raw text response into a ``TriageResult``,
    tolerant of a model that wraps its JSON in a markdown code fence or
    adds a sentence of preamble/trailing commentary — this never raises,
    regardless of how malformed the input is. An error class the model
    invented (not in ``valid_classes``) is silently dropped rather than
    accepted, since a class the taxonomy doesn't recognize has no
    ``candidate_levers`` for ``propose_lever`` to look up later anyway. A
    response with no JSON object at all, invalid JSON, or zero recognized
    error classes is marked ``needs_manual_review`` rather than returned as
    an empty-but-confident diagnosis.
    """
    match = _JSON_BLOCK_RE.search(raw_text or "")
    if not match:
        return TriageResult(needs_manual_review=True, parse_error="no JSON object found in model response")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return TriageResult(needs_manual_review=True, parse_error=f"invalid JSON: {exc}")
    if not isinstance(data, dict):
        return TriageResult(needs_manual_review=True, parse_error="parsed JSON is not an object")

    raw_classes = data.get("error_classes")
    if not isinstance(raw_classes, list):
        raw_classes = []
    error_classes = [c for c in raw_classes if isinstance(c, str) and c in valid_classes]
    dropped_unknown = [c for c in raw_classes if isinstance(c, str) and c not in valid_classes]

    target_spans = [s for s in (data.get("target_spans") or []) if isinstance(s, str)]
    expected_refs = [r for r in (data.get("expected_refs") or []) if isinstance(r, str)]
    severity = data.get("severity") if data.get("severity") in ("low", "medium", "high") else "unknown"
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    result = TriageResult(
        error_classes=error_classes, target_spans=target_spans, expected_refs=expected_refs,
        severity=severity, confidence=confidence,
    )
    if not error_classes:
        result.needs_manual_review = True
        result.parse_error = (
            f"model returned no recognized error class (dropped unknown: {dropped_unknown})"
            if dropped_unknown else "model returned no error classes"
        )
    elif confidence < _LOW_CONFIDENCE_THRESHOLD:
        result.needs_manual_review = True
    return result


def cross_check(
    result: TriageResult,
    source_refs: list[dict[str, Any]],
    pre_rerank_source_refs: list[dict[str, Any]] | None,
    funnel_layer: str | None,
) -> TriageResult:
    """Deterministic (no LLM) verification of the model's own diagnosis
    against this question's actual retrieval trace, mutating and
    returning ``result``. For each ``expected_ref`` the model extracted,
    checks whether it corresponds to a chunk that was actually retrieved
    (present in ``source_refs``, the post-rerank/final list, or in
    ``pre_rerank_source_refs`` — retrieved but then dropped before the
    final answer) by matching against that chunk's ``article_no``,
    ``source_code``, or ``structural_path``.

    ``diagnosis_confirmed`` is true when at least one ``expected_ref``
    verified this way. When the model extracted no ``expected_refs`` at
    all, there is nothing for this function to check against the run's
    own retrieval trace — ``diagnosis_confirmed`` then falls back to
    whether the funnel verdict itself is anything other than ``"ok"``
    (an "ok"-verdict question with no checkable ref genuinely has nothing
    to confirm, rather than being treated as confirmed by default).
    """
    result.funnel_layer = funnel_layer
    if not result.expected_refs:
        result.diagnosis_confirmed = funnel_layer is not None and funnel_layer != "ok"
        return result

    all_refs = list(source_refs) + list(pre_rerank_source_refs or [])
    known_article_nos = {r.get("article_no") for r in all_refs if r.get("article_no")}
    known_source_codes = {r.get("source_code") for r in all_refs if r.get("source_code")}
    known_paths = {r.get("structural_path") for r in all_refs if r.get("structural_path")}

    verified: dict[str, bool] = {
        ref: (ref in known_article_nos or ref in known_source_codes or ref in known_paths)
        for ref in result.expected_refs
    }
    result.verified_refs = verified
    result.diagnosis_confirmed = any(verified.values())
    return result


def propose_lever(result: TriageResult, taxonomy: list[dict[str, Any]]) -> dict[str, Any]:
    """Looks up each of ``result``'s error classes in ``taxonomy`` and
    returns a plain-language, human-reviewed proposal — never a fix
    applied automatically. The rule is "agent proposes, human decides"
    throughout: this function's entire output is a suggestion, not an
    action, and creating an actual pin/demote/promote remains a distinct,
    explicit step the reviewer takes afterward."""
    by_id = {c["id"]: c for c in taxonomy}
    levers: list[str] = []
    justifications: list[str] = []
    for cls in result.error_classes:
        entry = by_id.get(cls)
        if entry is None:
            continue
        entry_levers = entry.get("candidate_levers", [])
        for lv in entry_levers:
            if lv not in levers:
                levers.append(lv)
        if entry_levers:
            justifications.append(f"{cls}: {entry['definition']} → {', '.join(entry_levers)}")
    return {
        "primary_lever": levers[0] if levers else None,
        "alternative_levers": levers[1:],
        "justification": "; ".join(justifications),
    }
