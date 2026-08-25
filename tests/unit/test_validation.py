"""Unit tests — BasicValidator and InputValidator Protocol."""
from core.validation import BasicValidator, InputValidator


def test_basic_is_input_validator():
    assert isinstance(BasicValidator(), InputValidator)


def test_valid_text():
    v = BasicValidator()
    result = v.validate("What does article 5 require?")
    assert result.is_valid
    assert result.corrected_text == "What does article 5 require?"


def test_empty_string():
    v = BasicValidator()
    result = v.validate("")
    assert not result.is_valid
    assert result.error_message == "empty_input"


def test_whitespace_only():
    v = BasicValidator()
    result = v.validate("   ")
    assert not result.is_valid
    assert result.error_message == "empty_input"


def test_meaningless_symbols():
    v = BasicValidator()
    result = v.validate("!!! ??? ...")
    assert not result.is_valid
    assert result.error_message == "meaningless_input"


def test_too_short():
    v = BasicValidator()
    result = v.validate("ok")
    assert not result.is_valid
    assert result.error_message == "too_short"


def test_strips_whitespace():
    v = BasicValidator()
    result = v.validate("  a question  ")
    assert result.is_valid
    assert result.corrected_text == "a question"
