"""core/eval/triage.py — feedback-comment classification, parsing, and
deterministic cross-check. Pure logic only, tested
on plain fixtures — no LLM call anywhere in this module or these tests.
"""
from __future__ import annotations

import json

from core.eval.triage import build_triage_prompt, cross_check, parse_triage_response, propose_lever

_TAXONOMY = [
    {
        "id": "wrong_citation", "definition": "wrong article cited", "typical_layer": "generation_citation",
        "candidate_levers": ["fix_citation_mapping"],
    },
    {
        "id": "incomplete", "definition": "answer is missing something", "typical_layer": "retrieval",
        "candidate_levers": ["pin_missing_fragment", "rewrite"],
    },
    {
        "id": "wrong_document", "definition": "wrong document used", "typical_layer": "retrieval",
        "candidate_levers": ["pin_and_demote", "rewrite"],
    },
]
_VALID_CLASSES = frozenset(c["id"] for c in _TAXONOMY)


class TestBuildTriagePrompt:
    def test_prompt_includes_the_comment_and_every_class_definition(self) -> None:
        prompt = build_triage_prompt("the citation is wrong", _TAXONOMY)
        assert "the citation is wrong" in prompt
        assert "wrong_citation" in prompt
        assert "wrong article cited" in prompt


class TestParseTriageResponse:
    def test_valid_json_parses_cleanly(self) -> None:
        raw = json.dumps({
            "error_classes": ["wrong_citation"], "target_spans": ["article 5"],
            "expected_refs": ["art-10"], "severity": "medium", "confidence": 0.8,
        })
        result = parse_triage_response(raw, _VALID_CLASSES)
        assert result.error_classes == ["wrong_citation"]
        assert result.target_spans == ["article 5"]
        assert result.expected_refs == ["art-10"]
        assert result.severity == "medium"
        assert result.confidence == 0.8
        assert result.needs_manual_review is False
        assert result.parse_error is None

    def test_json_wrapped_in_markdown_code_fence_still_parses(self) -> None:
        raw = "Here is the answer:\n```json\n" + json.dumps({
            "error_classes": ["incomplete"], "confidence": 0.9,
        }) + "\n```\nThank you."
        result = parse_triage_response(raw, _VALID_CLASSES)
        assert result.error_classes == ["incomplete"]
        assert result.needs_manual_review is False

    def test_no_json_object_at_all_needs_manual_review(self) -> None:
        result = parse_triage_response("sorry, I cannot help", _VALID_CLASSES)
        assert result.needs_manual_review is True
        assert result.error_classes == []
        assert "no JSON object" in result.parse_error

    def test_invalid_json_needs_manual_review_not_a_crash(self) -> None:
        result = parse_triage_response("{not valid json at all}", _VALID_CLASSES)
        assert result.needs_manual_review is True
        assert "invalid JSON" in result.parse_error

    def test_empty_string_needs_manual_review_not_a_crash(self) -> None:
        result = parse_triage_response("", _VALID_CLASSES)
        assert result.needs_manual_review is True

    def test_unknown_error_class_is_dropped_not_accepted(self) -> None:
        raw = json.dumps({"error_classes": ["wrong_citation", "made_up_class"], "confidence": 0.9})
        result = parse_triage_response(raw, _VALID_CLASSES)
        assert result.error_classes == ["wrong_citation"]

    def test_only_unknown_classes_needs_manual_review_with_a_note(self) -> None:
        raw = json.dumps({"error_classes": ["totally_invented"], "confidence": 0.9})
        result = parse_triage_response(raw, _VALID_CLASSES)
        assert result.error_classes == []
        assert result.needs_manual_review is True
        assert "totally_invented" in result.parse_error

    def test_low_confidence_needs_manual_review_despite_a_valid_class(self) -> None:
        raw = json.dumps({"error_classes": ["wrong_citation"], "confidence": 0.1})
        result = parse_triage_response(raw, _VALID_CLASSES)
        assert result.error_classes == ["wrong_citation"]  # still returned...
        assert result.needs_manual_review is True          # ...just flagged

    def test_confidence_is_clamped_to_zero_one_range(self) -> None:
        raw = json.dumps({"error_classes": ["wrong_citation"], "confidence": 5.0})
        assert parse_triage_response(raw, _VALID_CLASSES).confidence == 1.0
        raw2 = json.dumps({"error_classes": ["wrong_citation"], "confidence": -3.0})
        assert parse_triage_response(raw2, _VALID_CLASSES).confidence == 0.0

    def test_missing_confidence_defaults_to_zero_and_flags_manual_review(self) -> None:
        raw = json.dumps({"error_classes": ["wrong_citation"]})
        result = parse_triage_response(raw, _VALID_CLASSES)
        assert result.confidence == 0.0
        assert result.needs_manual_review is True

    def test_invalid_severity_value_falls_back_to_unknown(self) -> None:
        raw = json.dumps({"error_classes": ["wrong_citation"], "severity": "catastrophic", "confidence": 0.9})
        assert parse_triage_response(raw, _VALID_CLASSES).severity == "unknown"


class TestCrossCheck:
    def test_expected_ref_found_in_source_refs_confirms_diagnosis(self) -> None:
        from core.eval.triage import TriageResult
        result = TriageResult(expected_refs=["art-10"])
        cross_check(result, source_refs=[{"article_no": "art-10"}], pre_rerank_source_refs=None, funnel_layer="retrieval")
        assert result.verified_refs == {"art-10": True}
        assert result.diagnosis_confirmed is True
        assert result.funnel_layer == "retrieval"

    def test_expected_ref_found_only_pre_rerank_still_confirms(self) -> None:
        from core.eval.triage import TriageResult
        result = TriageResult(expected_refs=["art-10"])
        cross_check(
            result, source_refs=[], pre_rerank_source_refs=[{"article_no": "art-10"}], funnel_layer="rerank",
        )
        assert result.verified_refs == {"art-10": True}
        assert result.diagnosis_confirmed is True

    def test_expected_ref_matching_nothing_is_not_confirmed(self) -> None:
        from core.eval.triage import TriageResult
        result = TriageResult(expected_refs=["art-99"])
        cross_check(result, source_refs=[{"article_no": "art-10"}], pre_rerank_source_refs=None, funnel_layer="ok")
        assert result.verified_refs == {"art-99": False}
        assert result.diagnosis_confirmed is False

    def test_matches_by_source_code_or_structural_path_too(self) -> None:
        from core.eval.triage import TriageResult
        result = TriageResult(expected_refs=["SRC005", "sec. 3"])
        cross_check(
            result,
            source_refs=[{"source_code": "SRC005"}, {"structural_path": "sec. 3"}],
            pre_rerank_source_refs=None, funnel_layer="ok",
        )
        assert result.verified_refs == {"SRC005": True, "sec. 3": True}
        assert result.diagnosis_confirmed is True

    def test_no_expected_refs_falls_back_to_funnel_layer_not_ok(self) -> None:
        from core.eval.triage import TriageResult
        result = TriageResult(expected_refs=[])
        cross_check(result, source_refs=[], pre_rerank_source_refs=None, funnel_layer="generation")
        assert result.diagnosis_confirmed is True

    def test_no_expected_refs_and_ok_funnel_is_not_confirmed(self) -> None:
        from core.eval.triage import TriageResult
        result = TriageResult(expected_refs=[])
        cross_check(result, source_refs=[], pre_rerank_source_refs=None, funnel_layer="ok")
        assert result.diagnosis_confirmed is False


class TestProposeLever:
    def test_single_class_proposes_its_own_levers(self) -> None:
        from core.eval.triage import TriageResult
        result = TriageResult(error_classes=["wrong_citation"])
        proposal = propose_lever(result, _TAXONOMY)
        assert proposal["primary_lever"] == "fix_citation_mapping"
        assert proposal["alternative_levers"] == []
        assert "wrong_citation" in proposal["justification"]

    def test_multiple_classes_deduplicate_shared_levers(self) -> None:
        from core.eval.triage import TriageResult
        result = TriageResult(error_classes=["incomplete", "wrong_document"])
        proposal = propose_lever(result, _TAXONOMY)
        # "rewrite" is a candidate lever for both classes — must appear once.
        all_levers = [proposal["primary_lever"], *proposal["alternative_levers"]]
        assert all_levers.count("rewrite") == 1

    def test_unknown_class_is_ignored_not_a_crash(self) -> None:
        from core.eval.triage import TriageResult
        result = TriageResult(error_classes=["not_in_taxonomy"])
        proposal = propose_lever(result, _TAXONOMY)
        assert proposal["primary_lever"] is None
        assert proposal["justification"] == ""

    def test_no_error_classes_proposes_nothing(self) -> None:
        from core.eval.triage import TriageResult
        proposal = propose_lever(TriageResult(error_classes=[]), _TAXONOMY)
        assert proposal == {"primary_lever": None, "alternative_levers": [], "justification": ""}
