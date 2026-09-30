"""One copy of every fragment text, addressed by what it says.

A run stores the fragments it retrieved, and a fragment's text used to travel
with every reference to it. Measured over the 142 runs stored when this was
written: 137 MiB of text in a 217 MiB store, of which 16 MiB were distinct.
A text now lives once in a store of its own and a reference carries its
identifier.

The identifier is taken from the text and never from the fragment's own
identifier, because the second does not determine the first: 26 fragment
identifiers in those runs stood for more than one text, and a store keyed on
them would have shown one text where the model had read another.

What stays on a reference is what the checks read without the text: the
identifier, which tells two texts apart, and the length after stripping
surrounding whitespace, which is what a check for fragments that carry only a
heading measures. Pure: no I/O.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

#: Hexadecimal characters of SHA-256 kept. 128 bits: two different texts
#: sharing one is not something any store this platform keeps will see.
TEXT_ID_CHARS = 32


def text_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:TEXT_ID_CHARS]


def stripped_chars(text: str) -> int:
    return len(text.strip())


def text_key(ref: dict[str, Any]) -> str | None:
    """The identifier of a reference's text, whichever shape it was stored in.

    A run written before the store carries the text itself, and its
    identifier is computed; a reference that carries neither has no text to
    speak of, which is different from an empty one.
    """
    if ref.get("text_id"):
        return str(ref["text_id"])
    if isinstance(ref.get("chunk_text"), str):
        return text_id(ref["chunk_text"])
    return None


def text_chars(ref: dict[str, Any]) -> int | None:
    """The stripped length of a reference's text, in either shape."""
    if isinstance(ref.get("text_chars"), int):
        return int(ref["text_chars"])
    if isinstance(ref.get("chunk_text"), str):
        return stripped_chars(ref["chunk_text"])
    return None


def _refs(holder: dict[str, Any], fields: Iterable[str]) -> Iterable[dict[str, Any]]:
    for field in fields:
        for ref in holder.get(field) or []:
            if isinstance(ref, dict):
                yield ref


def externalise(data: dict[str, Any], fields: Iterable[str]) -> tuple[dict[str, Any], dict[str, str]]:
    """The run with every text replaced by its identifier, and the texts.

    Copied and never mutated: the caller may still need the run as it was,
    and a file written from it when the store could not be reached is that.
    """
    fields = tuple(fields)
    texts: dict[str, str] = {}
    questions = []
    for question in data.get("question_results") or []:
        rewritten = dict(question)
        for field in fields:
            refs = question.get(field)
            if not refs:
                continue
            out = []
            for ref in refs:
                if not isinstance(ref, dict) or not isinstance(ref.get("chunk_text"), str):
                    out.append(ref)
                    continue
                text = ref["chunk_text"]
                key = text_id(text)
                texts[key] = text
                shaped = {k: v for k, v in ref.items() if k != "chunk_text"}
                shaped["text_id"] = key
                shaped["text_chars"] = stripped_chars(text)
                out.append(shaped)
            rewritten[field] = out
        questions.append(rewritten)
    return {**data, "question_results": questions}, texts


def text_ids_in(holders: Iterable[dict[str, Any]], fields: Iterable[str]) -> set[str]:
    """Every text identifier the given questions or windows refer to."""
    fields = tuple(fields)
    return {str(ref["text_id"]) for holder in holders for ref in _refs(holder, fields)
            if ref.get("text_id")}


def internalise(data: dict[str, Any], fields: Iterable[str], texts: dict[str, str]) -> dict[str, Any]:
    """The run with each identifier's text put back beside it.

    A reference whose text the store no longer holds is marked and given an
    empty text, and never left without the key: a reader that met an absent
    text would render nothing and say nothing, which is the same as saying
    the fragment was empty.
    """
    fields = tuple(fields)
    questions = []
    for question in data.get("question_results") or []:
        rewritten = dict(question)
        for field in fields:
            refs = question.get(field)
            if not refs:
                continue
            out = []
            for ref in refs:
                if isinstance(ref, dict) and ref.get("text_id") and "chunk_text" not in ref:
                    key = str(ref["text_id"])
                    if key in texts:
                        ref = {**ref, "chunk_text": texts[key]}
                    else:
                        ref = {**ref, "chunk_text": "", "text_missing": True}
                out.append(ref)
            rewritten[field] = out
        questions.append(rewritten)
    return {**data, "question_results": questions}
