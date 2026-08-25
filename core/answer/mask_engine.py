from __future__ import annotations

import re
import string
from typing import Any

# Source paths often arrive as a raw structural_path, e.g.
# "document/article[Article 774]": the bracketed text is the
# human-readable label a structure_parser produced at ingest time; the
# rest is internal addressing that shouldn't reach the user-facing answer.
# Mirrors core/citation.py's _label_for(), generically (no domain terms
# here — whatever a given corpus's parser puts in brackets, or the parser
# emits no brackets at all and the path is used as-is, e.g. plain labels
# like "§1" already used in tests).
_LABEL_RE = re.compile(r"\[([^\]]+)\]")


def _clean_source_label(path: str) -> str:
    match = _LABEL_RE.search(path)
    return match.group(1) if match else path


class AnswerTemplate:
    """Renders an answer using a mask loaded from config.

    A mask is a dict with:
      - ``template``: str with ``{answer}``, ``{sources}``, ``{disclaimer}`` placeholders
      - ``disclaimer``: optional str appended after the answer (domain-supplied)
      - ``sources_format``: ``"numbered"`` | ``"inline"`` | ``"none"``
    """

    def __init__(self, question_type: str, mask: dict[str, Any]) -> None:
        self.question_type = question_type
        self._template: str = mask.get("template", "{answer}")
        self._disclaimer: str = mask.get("disclaimer", "")
        self._sources_format: str = mask.get("sources_format", "none")

    def render(
        self,
        answer_text: str,
        source_paths: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        sources_str = self._format_sources(source_paths or [])
        variables = {
            "answer": answer_text,
            "sources": sources_str,
            "disclaimer": self._disclaimer,
            **(extra or {}),
        }
        # Only substitute known keys; leave unknown placeholders as-is.
        result = self._safe_format(self._template, variables)
        return result.strip()

    def _format_sources(self, paths: list[str]) -> str:
        if not paths or self._sources_format == "none":
            return ""
        labels = [_clean_source_label(p) for p in paths]
        if self._sources_format == "numbered":
            return "\n".join(f"{i+1}. {p}" for i, p in enumerate(labels))
        return "; ".join(labels)

    @staticmethod
    def _safe_format(template: str, variables: dict[str, Any]) -> str:
        result = []
        formatter = string.Formatter()
        for literal, field_name, _, _ in formatter.parse(template):
            result.append(literal)
            if field_name is not None:
                result.append(str(variables.get(field_name, f"{{{field_name}}}")))
        return "".join(result)


class MaskEngine:
    """Manages a set of AnswerTemplates keyed by question_type."""

    def __init__(self, masks: dict[str, dict[str, Any]]) -> None:
        self._templates: dict[str, AnswerTemplate] = {
            qt: AnswerTemplate(qt, mask) for qt, mask in masks.items()
        }
        self._fallback = AnswerTemplate("_fallback", {"template": "{answer}"})

    def get(self, question_type: str) -> AnswerTemplate:
        return self._templates.get(question_type, self._fallback)

    def render(
        self,
        question_type: str,
        answer_text: str,
        source_paths: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        return self.get(question_type).render(answer_text, source_paths, extra)

    @classmethod
    def from_dict(cls, masks: dict[str, dict[str, Any]]) -> MaskEngine:
        return cls(masks)
