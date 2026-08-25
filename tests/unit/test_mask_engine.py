"""Unit tests — MaskEngine and AnswerTemplate."""
from core.answer.mask_engine import MaskEngine

MASKS = {
    "closed": {
        "template": "{answer}\n\n{disclaimer}",
        "disclaimer": "This answer is for reference only.",
        "sources_format": "none",
    },
    "open": {
        "template": "{answer}\n\nSources:\n{sources}",
        "disclaimer": "",
        "sources_format": "numbered",
    },
    "navigational": {
        "template": "See: {sources}\n\n{answer}",
        "disclaimer": "",
        "sources_format": "inline",
    },
}


def test_render_closed_no_sources():
    engine = MaskEngine.from_dict(MASKS)
    result = engine.render("closed", "Yes, the rule applies.")
    assert "Yes, the rule applies." in result
    assert "for reference only" in result


def test_render_open_numbered_sources():
    engine = MaskEngine.from_dict(MASKS)
    result = engine.render("open", "A detailed answer.", ["§1", "§2"])
    assert "1. §1" in result
    assert "2. §2" in result


def test_render_navigational_inline_sources():
    engine = MaskEngine.from_dict(MASKS)
    result = engine.render("navigational", "Found.", ["Section 3", "Section 5"])
    assert "Section 3; Section 5" in result


def test_fallback_for_unknown_type():
    engine = MaskEngine.from_dict(MASKS)
    result = engine.render("unknown_type", "An answer.")
    assert "An answer." in result


def test_extra_variables():
    masks = {"custom": {"template": "{answer} [{label}]", "sources_format": "none"}}
    engine = MaskEngine.from_dict(masks)
    result = engine.render("custom", "Some text.", extra={"label": "TEST"})
    assert "[TEST]" in result


def test_numbered_sources_strip_raw_structural_path_to_bracket_label():
    """Real bug observed live: source_paths from core/pipeline.py are raw
    structural_path strings like "document/article[Article 774]" — only the
    bracketed label should reach the user-facing answer, not the internal
    addressing prefix."""
    engine = MaskEngine.from_dict(MASKS)
    result = engine.render(
        "open", "An answer.",
        ["document/article[Article 774]", "document/article[Article 655. Ending an agreement]"],
    )
    assert "1. Article 774" in result
    assert "2. Article 655. Ending an agreement" in result
    assert "document/article" not in result


def test_inline_sources_strip_raw_structural_path_to_bracket_label():
    engine = MaskEngine.from_dict(MASKS)
    result = engine.render("navigational", "Found.", ["document/article[Article 5]"])
    assert "Article 5" in result
    assert "document/article" not in result


def test_sources_without_brackets_are_used_as_is():
    """Plain labels with no structural_path-style brackets (e.g. a
    different domain's parser, or the pre-existing "§1" test fixture
    style) must pass through unchanged."""
    engine = MaskEngine.from_dict(MASKS)
    result = engine.render("open", "An answer.", ["§1", "Page 12"])
    assert "1. §1" in result
    assert "2. Page 12" in result
