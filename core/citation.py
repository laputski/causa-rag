"""Programmatic citation labels — domain-agnostic.

Generators (adapters/ollama_generator.py, adapters/vllm.py) are unreliable
at transcribing exact citation numbers into free text: a model confused the
prompt's "Фрагмент N" fragment marker with the real citation number, and on
other questions invented a number that wasn't among the retrieved chunks at
all, despite retrieval itself having found the right chunk
(retrieval_recall_at_k=1.0 in both observed cases) — the correct citation
was already sitting in source_refs, just never reliably reaching the
generated text.

compute_citation_labels() sidesteps free-text transcription entirely: it
reads the label straight from SourceRef.structural_path, which a
domain-specific structure_parser (under domain_packs/<pack>/) already
produced at ingest time. That label's exact wording is whatever the parser
for a given corpus chooses to emit — this module never hard-codes a format
string, it only extracts whatever bracketed label is already there, or
falls back to doc_id when there is none, so it works unchanged for corpora
with no such structure at all.
"""
from __future__ import annotations

import re

from core.models import SourceRef

_LABEL_RE = re.compile(r"\[([^\]]+)\]")
# Found live: the model sometimes code-switches mid-word, emitting "Фragment 7"
# (Cyrillic Ф plus Latin "ragment") instead of the correct "Фрагмент 7". Not
# a visual homoglyph attack, just a generation artifact — matching the
# Latin transliteration too (with an optional single leading character of
# either script) catches this without trying to enumerate every possible
# typo.
_FRAGMENT_MARKER_RE = re.compile(r"(?:Фрагмент|[ФF]?ragment)\s+(\d+)", re.IGNORECASE)
# The plural, and the list that comes with it. Found live on the proving
# ground: asked about four instruments at once the model wrote "(Фрагменты 2,
# 3, 4, 5)", which the singular pattern above cannot match, because
# "Фрагмент" is a prefix of "Фрагменты" and the \s+ that follows meets a
# letter. The
# marker reached the user verbatim, which substitute_fragment_markers' own
# docstring calls structurally impossible. It was impossible only for the
# form the model happened not to use.
_FRAGMENT_LIST_RE = re.compile(
    r"(?:Фрагмент\w*|[ФF]?ragments?)\s+(\d+(?:\s*(?:,|и|and|&)\s*\d+)+)", re.IGNORECASE)
_LIST_NUMBER_RE = re.compile(r"\d+")
# Compound numbering (e.g. "210.5", "16-9") — same convention already used
# for this elsewhere in the codebase (core/chunking/structure_aware.py's
# _BARE_NUMERAL_RE, a domain pack's structure parser regex).
_NUMBER_RE = re.compile(r"\d+(?:[.\-]\d+)*")
_ANY_DIGIT_RE = re.compile(r"\d")


def _label_for(ref: SourceRef) -> str:
    match = _LABEL_RE.search(ref.structural_path or "")
    if match:
        return match.group(1)
    return ref.structural_path or ref.doc_id


def _bracketed_label(ref: SourceRef) -> str | None:
    """Like _label_for, but returns None (not a doc_id fallback) when the
    corpus has no structure_parser-produced bracketed label — used by
    citation_number_coverage() below, where falling back to doc_id would
    extract spurious digit fragments out of a UUID and silently corrupt
    the metric instead of correctly reporting "not applicable" for
    corpora/leaves with no extractable structural number at all."""
    match = _LABEL_RE.search(ref.structural_path or "")
    return match.group(1) if match else None


def compute_citation_labels(answer_text: str, source_refs: list[SourceRef]) -> list[str]:
    """Labels to show as the *real* citation, computed from retrieval
    metadata rather than trusted from generated text.

    If the model referenced one or more "Фрагмент N" fragment indices (the
    label the prompt uses, core/prompt_store.py render()), returns the
    label for each distinct fragment referenced, in the order first
    mentioned. Otherwise falls back to the single top-ranked source
    (source_refs[0]) — the chunk the pipeline considered most relevant.
    Returns [] when there's nothing to cite from (e.g. a refusal with no
    retrieved chunks).
    """
    # NOTE: the "Фрагмент N" marker text itself is generic (the prompt's own
    # fragment-index labeling, core/prompt_store.py render() — not a
    # domain-specific citation format), so matching it here doesn't
    # reintroduce domain knowledge into core/.
    if not source_refs:
        return []

    seen: set[int] = set()
    labels: list[str] = []
    for m in _FRAGMENT_MARKER_RE.finditer(answer_text):
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(source_refs) and idx not in seen:
            seen.add(idx)
            labels.append(_label_for(source_refs[idx]))

    if labels:
        return labels
    return [_label_for(source_refs[0])]


def _flexible_number_pattern(number: str) -> re.Pattern[str]:
    """Builds a regex that finds `number` (e.g. "210.5") in free text as a
    whole token, tolerating "." / "-" / "," as the separator between digit
    groups — the source label and the model's own transcription don't
    necessarily agree on which punctuation mark to use for a compound
    number. Word-boundary lookaround (not part of a longer digit run)
    prevents "5" from matching inside "2025"."""
    parts = re.split(r"[.\-]", number)
    body = r"[.,\-]".join(re.escape(p) for p in parts)
    return re.compile(rf"(?<!\d){body}(?!\d)")


def citation_number_coverage(
    answer_text: str, source_refs: list[SourceRef], article_refs: list[str] | None = None,
) -> float | None:
    """Domain-agnostic citation-correctness signal: does the answer text
    contain the actual structural number(s) retrieval found, rather than
    trusting the model to have transcribed them correctly?

    Deliberately does NOT try to detect "this looks like a citation" in
    the answer text via any noun/format heuristic (e.g. an article number,
    "Article N") — that would just be guessing one domain's citation
    grammar. Instead it flips the problem: the small set of candidate
    numbers is already known from source_refs, so this only checks
    whether each one's digits occur anywhere in the answer text at all.

    ``article_refs`` (ground-truth ref-id strings built by
    ``core/eval/retrieval_metrics.py#extract_ref_id`` — same convention as
    ``core/eval/semantic_metrics.py:grounded_in_correct_source``, must stay
    in sync with it) restricts the candidate set to only the chunk(s)
    actually correct for this question, not every chunk retrieval happened
    to return. Without this filter the metric divides by ALL distinct
    numbers among, say, top_k=10 retrieved chunks — a single-citation
    answer that correctly cites the one right source then structurally caps
    near 1/(distinct numbers in top_k), making a fully-correct citation read
    as "mostly wrong" (found live: 0.125-0.25 on run 9da7259a despite every
    sampled citation matching its question by content). When
    ``article_refs`` is omitted (back-compat / no ground truth available),
    falls back to using every source_ref.

    Found live (a second time): the match itself used to be a hardcoded
    "{source_code}/{article_no}" lookup, which a corpus with no external
    document-code numbering scheme never satisfies (source_code/article_no
    are always absent there) — every candidate number silently vanished
    (None) for every question of every such run, not because nothing was
    cited correctly but because the match could never succeed at all. Now
    built via extract_ref_id, same as grounded_in_correct_source's fix.

    Returns the fraction of distinct candidate numbers found in the text
    (1.0 = every retrieved label's number appears somewhere in the
    answer — including multi-reference answers, not just the top hit).
    Returns None — not 0.0 — when there are no candidate numbers to check
    against at all: either no source_ref has an extractable bracketed
    structural label (e.g. a corpus with no structure_parser/flat
    chunking), or (when article_refs is given) none of the retrieved
    chunks actually match the ground truth (recall_at_k=0 — same
    None-not-0.0 convention as grounded_in_correct_source, since a 0.0
    here would misleadingly read as "wrong citation" rather than "nothing
    relevant was even retrieved").

    Known limitation: a candidate number can coincidentally appear in the
    text for an unrelated reason (an amount, a date, a quantity) — this
    is a coverage signal, not a verified-citation precision metric.
    """
    if article_refs is not None:
        from core.eval.retrieval_metrics import extract_ref_id

        expected = set(article_refs)
        relevant_refs = [
            ref for ref in source_refs
            if extract_ref_id(ref.model_dump()) in expected
        ]
    else:
        relevant_refs = source_refs

    candidate_numbers: set[str] = set()
    for ref in relevant_refs:
        label = _bracketed_label(ref)
        if label:
            candidate_numbers |= set(_NUMBER_RE.findall(label))

    if not candidate_numbers:
        return None

    matched = sum(
        1 for number in candidate_numbers
        if _flexible_number_pattern(number).search(answer_text)
    )
    return matched / len(candidate_numbers)


def substitute_fragment_markers(answer_text: str, source_refs: list[SourceRef]) -> str:
    """Replaces every "Фрагмент N" occurrence in the answer text with the
    real structural label for that fragment — used instead of trusting the
    model to write the real citation number itself (the active prompt now
    instructs the model to use ONLY the positional "Фрагмент N" marker,
    the same "cite by index, not by content" pattern production systems
    use — Perplexity/Bing render a small bracketed index and resolve it to
    the real source in the UI/backend, never asking the model to reproduce
    the source's own numbering scheme).

    Production motivation: raw "(Фрагмент 1)" markers were leaking
    verbatim into the user-facing answer in ~50% of one real run's
    generations — confusing on its own even before considering whether the
    index pointed at the right source. This makes that leak structurally
    impossible: every marker is substituted before the answer is returned,
    so the user never sees the internal "Фрагмент N" wording at all.

    An out-of-range index (model invented a fragment number beyond what
    was retrieved) is removed rather than left as a dangling raw marker —
    same reasoning as compute_citation_labels' out-of-range handling.

    Both the singular marker and the plural list are substituted. The claim
    of impossibility above was written when only the singular was handled,
    and a run on the proving ground found the plural reaching the user
    untouched; the claim is now true of the form that was observed as well
    as the form that was anticipated.
    """
    def _replace(match: re.Match[str]) -> str:
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(source_refs):
            return _label_for(source_refs[idx])
        return ""

    def _replace_list(match: re.Match[str]) -> str:
        labels = []
        for number in _LIST_NUMBER_RE.findall(match.group(1)):
            idx = int(number) - 1
            if 0 <= idx < len(source_refs):
                labels.append(_label_for(source_refs[idx]))
        return ", ".join(labels)

    # The list first: the singular pattern would otherwise consume the head of
    # "Фрагменты 2, 3" as far as it can and leave the rest of the list
    # standing beside a substituted label.
    result = _FRAGMENT_LIST_RE.sub(_replace_list, answer_text)
    result = _FRAGMENT_MARKER_RE.sub(_replace, result)
    # Clean up now-empty "()" left behind when an out-of-range marker was
    # the sole content of a parenthetical, the dangling space before
    # punctuation that leaves, and any doubled whitespace.
    result = re.sub(r"\(\s*\)", "", result)
    result = re.sub(r"\s+([.,;:!?])", r"\1", result)
    return re.sub(r"[ \t]{2,}", " ", result).strip()


def answer_has_any_number(answer_text: str) -> bool:
    """Weak, format-free signal for "did the model write anything
    number-like at all" — used alongside citation_number_coverage to tell
    apart "no citation attempted" from "citation attempted but wrong"
    (both currently look the same as coverage=0.0 on their own)."""
    return bool(_ANY_DIGIT_RE.search(answer_text))
