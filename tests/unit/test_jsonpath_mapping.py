"""adapters/jsonpath_mapping.py (tier 2, declarative mapping)."""
from __future__ import annotations

from adapters.jsonpath_mapping import apply_response_mapping, render_request_template


def test_exact_placeholder_preserves_native_type():
    template = {"q": "{{query}}", "k": "{{top_k}}", "f": "{{filters}}", "t": "{{trace_id}}"}
    rendered = render_request_template(template, query="a question", top_k=5, filters={"a": 1}, trace_id="tr-1")

    assert rendered == {"q": "a question", "k": 5, "f": {"a": 1}, "t": "tr-1"}
    assert isinstance(rendered["k"], int)


def test_embedded_placeholder_is_string_interpolated():
    template = {"prompt": "Find an answer to: {{query}} (top {{top_k}})"}
    rendered = render_request_template(template, query="a question", top_k=3, filters={}, trace_id="t")

    assert rendered["prompt"] == "Find an answer to: a question (top 3)"


def test_nested_dict_and_list_are_rendered_recursively():
    template = {"outer": {"inner": ["{{query}}", {"k": "{{top_k}}"}]}}
    rendered = render_request_template(template, query="q", top_k=2, filters={}, trace_id="t")

    assert rendered == {"outer": {"inner": ["q", {"k": 2}]}}


def test_non_placeholder_values_pass_through_unchanged():
    template = {"static": "unchanged", "n": 42, "flag": True}
    rendered = render_request_template(template, query="q", top_k=1, filters={}, trace_id="t")

    assert rendered == {"static": "unchanged", "n": 42, "flag": True}


def test_apply_response_mapping_extracts_answer_and_sources():
    raw = {
        "result": {
            "text": "an answer",
            "docs": [
                {"id": "d1", "content": "text1", "meta": {"code": "HK1", "art": "5"}},
                {"id": "d2", "content": "text2", "meta": {"code": "HK2", "art": "7"}},
            ],
        }
    }
    mapping = {
        "answer": "$.result.text",
        "sources": "$.result.docs",
        "source_doc_id": "$.id",
        "source_text": "$.content",
        "source_code": "$.meta.code",
        "source_article_no": "$.meta.art",
    }

    parsed = apply_response_mapping(raw, mapping)

    assert parsed["answer"] == "an answer"
    assert len(parsed["sources"]) == 2
    assert parsed["sources"][0]["doc_id"] == "d1"
    assert parsed["sources"][0]["chunk_text"] == "text1"
    assert parsed["sources"][0]["source_code"] == "HK1"
    assert parsed["sources"][0]["article_no"] == "5"
    assert parsed["sources"][1]["doc_id"] == "d2"


def test_missing_answer_path_degrades_to_empty_string():
    parsed = apply_response_mapping({"foo": "bar"}, {"answer": "$.nope"})
    assert parsed["answer"] == ""


def test_missing_sources_path_degrades_to_empty_list():
    parsed = apply_response_mapping({"foo": "bar"}, {"answer": "$.foo"})
    assert parsed["sources"] == []


def test_invalid_jsonpath_expression_degrades_to_none_not_raise():
    """A misconfigured JSONPath is a mapping bug, not a platform crash —
    surfaces as missing/empty data in the probe response."""
    parsed = apply_response_mapping({"foo": "bar"}, {"answer": "$$$not valid", "sources": "$$$also invalid"})
    assert parsed["answer"] == ""
    assert parsed["sources"] == []


def test_source_item_missing_field_defaults_to_safe_value():
    raw = {"docs": [{"id": "d1"}]}
    mapping = {"sources": "$.docs", "source_doc_id": "$.id", "source_score": "$.score"}

    parsed = apply_response_mapping(raw, mapping)

    assert parsed["sources"][0]["doc_id"] == "d1"
    assert parsed["sources"][0]["score"] == 0.0
    assert parsed["sources"][0]["chunk_text"] == ""
