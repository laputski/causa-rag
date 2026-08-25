"""Unit tests — RefusalPolicy and grounding guard."""
from core.models import GroundingResult
from core.refusal import RefusalPolicy, ThresholdRefusalPolicy, apply_grounding_guard


def _grounding(is_grounded: bool, confidence: float) -> GroundingResult:
    return GroundingResult(is_grounded=is_grounded, confidence=confidence)


def test_threshold_policy_is_refusal_policy():
    assert isinstance(ThresholdRefusalPolicy(), RefusalPolicy)


def test_refuses_when_not_grounded():
    policy = ThresholdRefusalPolicy(min_confidence=0.5)
    assert policy.should_refuse(_grounding(is_grounded=False, confidence=0.3))


def test_refuses_when_low_confidence():
    policy = ThresholdRefusalPolicy(min_confidence=0.6)
    assert policy.should_refuse(_grounding(is_grounded=True, confidence=0.4))


def test_accepts_when_grounded_and_high_confidence():
    policy = ThresholdRefusalPolicy(min_confidence=0.5)
    assert not policy.should_refuse(_grounding(is_grounded=True, confidence=0.9))


def test_refusal_answer_is_refused():
    policy = ThresholdRefusalPolicy()
    answer = policy.build_refusal("low_confidence", "test")
    assert answer.refused
    assert answer.refusal_reason == "low_confidence"
    assert answer.text


def test_grounding_guard_produces_refusal():
    policy = ThresholdRefusalPolicy(min_confidence=0.5)
    grounding = _grounding(is_grounded=False, confidence=0.1)
    answer = apply_grounding_guard("some text", grounding, policy, "a query")
    assert answer.refused


def test_grounding_guard_passes_good_answer():
    policy = ThresholdRefusalPolicy(min_confidence=0.5)
    grounding = _grounding(is_grounded=True, confidence=0.95)
    answer = apply_grounding_guard("a good answer", grounding, policy, "a query")
    assert not answer.refused
    assert answer.text == "a good answer"
