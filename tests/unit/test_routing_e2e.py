"""The routing chain end to end: validation, classification, answer mask.

All of it on stubs, with no live services. The chain hangs off the domain
pack: the pack decides a question's type, the pack supplies the mask for that
type, and the platform only joins the two. The platform knows nothing about
the types themselves, which is the property these tests exist to hold.

The subject is the technical-manuals pack, the one that ships with the
platform and matches its own demo Realm.
"""
from core.models import QueryRequest
from core.routing import NaiveRoutePolicy
from core.validation import BasicValidator
from domain_packs.manuals import load_mask_engine
from domain_packs.manuals.routing import ManualsRoutePolicy


# The questions below are in Russian on purpose. `classify_question_type` in
# domain_packs/manuals/routing.py matches literal keywords, and that pack's
# keywords are Russian, so an English question would match none of them and
# every type assertion here would pass while checking nothing.
def _run_pipeline(text: str, context_chunks: list, policy=None):
    """Replays the routed pipeline: validate, classify, render the mask."""
    validator = BasicValidator()
    val_result = validator.validate(text)
    if not val_result.is_valid:
        return {"refused": True, "reason": val_result.error_message, "answer": None}

    pol = policy or ManualsRoutePolicy()
    request = QueryRequest(text=val_result.corrected_text)
    decision = pol.classify(request)

    engine = load_mask_engine()
    rendered = engine.render(
        decision.question_type,
        "An answer based on the context.",
        list(context_chunks),
    )

    return {
        "refused": False,
        "mode": decision.mode,
        "question_type": decision.question_type,
        "has_context": bool(context_chunks),
        "answer": rendered,
    }


def test_e2e_closed_question_with_context():
    result = _run_pipeline("поддерживает ли система DICOM?", ["Раздел 3.1"])
    assert not result["refused"]
    assert result["has_context"]
    assert result["question_type"] == "closed"
    assert result["answer"]


def test_e2e_empty_query_refused():
    result = _run_pipeline("", [])
    assert result["refused"]
    assert result["reason"] == "empty_input"


def test_e2e_meaningless_query_refused():
    result = _run_pipeline("!!! ???", [])
    assert result["refused"]


def test_e2e_no_context_still_renders():
    """An empty context is a retrieval outcome, and the mask engine has to
    produce something a reader can act on rather than raising."""
    result = _run_pipeline("что-то непонятное", [])
    assert not result["refused"]
    assert not result["has_context"]
    assert result["answer"]


def test_e2e_procedure_question():
    result = _run_pipeline("как заменить детектор?", ["4.2.1 Замена детектора"])
    assert not result["refused"]
    assert result["question_type"] == "procedure"


def test_e2e_safety_question_outranks_a_procedure_wording():
    # "How do I replace a part while it is live" is a safety question even
    # though it opens like a procedure, and the mask has to lead with the
    # warning and put the source after it.
    result = _run_pipeline("как заменить деталь, если это опасно?", ["5 Техника безопасности"])
    assert result["question_type"] == "safety"
    assert "безопасн" in result["answer"].lower()


def test_e2e_an_unrecognised_question_still_gets_a_mask():
    result = _run_pipeline("что такое индекс отклонения?", ["Глоссарий"])
    assert result["question_type"] == "open"
    assert result["answer"]


def test_e2e_the_platform_s_own_policy_needs_no_pack():
    # The platform can answer with no pack at all. The naive policy is its
    # own, and the chain still holds together on it.
    result = _run_pipeline("как заменить детектор?", ["4.2.1"], policy=NaiveRoutePolicy())
    assert not result["refused"]
    assert result["answer"]
