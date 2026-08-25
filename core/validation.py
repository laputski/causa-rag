from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class ValidationResult:
    is_valid: bool
    error_message: str = ""
    corrected_text: str = ""
    suggestions: list[str] = field(default_factory=list)


@runtime_checkable
class InputValidator(Protocol):
    validator_id: str

    def validate(self, text: str) -> ValidationResult:
        ...


@runtime_checkable
class SpellChecker(Protocol):
    checker_id: str

    def check(self, text: str) -> list[str]:
        """Return list of misspelled tokens."""
        ...

    def suggest(self, token: str) -> list[str]:
        """Return correction suggestions for a single token."""
        ...


_MEANINGLESS = re.compile(r"^[\W\d_\s]+$", re.UNICODE)


class BasicValidator:
    """Validates that input is non-empty and contains meaningful text."""

    validator_id = "basic"

    def validate(self, text: str) -> ValidationResult:
        stripped = text.strip()
        if not stripped:
            return ValidationResult(is_valid=False, error_message="empty_input")
        if _MEANINGLESS.match(stripped):
            return ValidationResult(is_valid=False, error_message="meaningless_input")
        if len(stripped) < 3:
            return ValidationResult(is_valid=False, error_message="too_short")
        return ValidationResult(is_valid=True, corrected_text=stripped)
