"""Contract tests — RoutePolicy Protocol compliance."""
from core.models import QueryRequest
from core.routing import NaiveRoutePolicy, RouteDecision, RoutePolicy


def _make_request(**kwargs) -> QueryRequest:
    return QueryRequest(text="test", **kwargs)


def test_naive_is_route_policy():
    assert isinstance(NaiveRoutePolicy(), RoutePolicy)


def test_naive_returns_route_decision():
    policy = NaiveRoutePolicy()
    decision = policy.classify(_make_request())
    assert isinstance(decision, RouteDecision)


def test_naive_passes_through_mode():
    policy = NaiveRoutePolicy()
    decision = policy.classify(_make_request(mode="regulations", question_type="closed"))
    assert decision.mode == "regulations"
    assert decision.question_type == "closed"
    assert decision.pipeline_id == "naive"


def test_naive_defaults_when_empty():
    policy = NaiveRoutePolicy()
    decision = policy.classify(_make_request())
    assert decision.mode == "default"
    assert decision.question_type == "open"
