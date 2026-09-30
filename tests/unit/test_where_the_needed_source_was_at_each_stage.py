"""Where the source a question needed was, at each stage of its retrieval."""
from __future__ import annotations

from core.eval.stages import stages_of


def _ref(n: int, doc: str = "doc") -> dict:
    """A fragment whose reference id is `base/<doc>` and whose id is unique."""
    return {"chunk_id": f"{doc}-{n}", "source_code": "base", "article_no": doc}


def _others(count: int, start: int = 0) -> list[dict]:
    return [_ref(i, f"other{i}") for i in range(start, start + count)]


def _by_name(result: dict) -> dict:
    return {s["stage"]: s for s in result["stages"]}


def test_a_source_the_reranker_dropped_is_lost_at_the_reranker() -> None:
    gold = _ref(1, "07")
    window = _others(5) + [gold] + _others(4, 10)
    question = {
        "expected_refs": ["base/07"],
        "dense_source_refs": _others(2) + [gold],
        "sparse_source_refs": _others(3, 20),
        "candidate_source_refs": window, "pre_rerank_source_refs": window,
        "source_refs": _others(5),
    }
    result = stages_of(question)
    stages = _by_name(result)
    assert [s["stage"] for s in result["stages"]] == ["dense", "sparse", "fusion", "rerank", "context"]
    assert (stages["dense"]["rank"], stages["sparse"]["rank"], stages["fusion"]["rank"]) == (3, None, 6)
    assert result["lost_at"] == "rerank" and result["found"] is True
    # The reranker's output is what reached the model, so the two are one list.
    assert stages["context"]["same_as"] == "rerank"


def test_a_source_one_half_found_and_the_merge_dropped_is_lost_at_the_merge() -> None:
    """The case the halves were recorded for. Without them this question read
    as one retrieval never found the source for."""
    gold = _ref(1, "07")
    question = {
        "expected_refs": ["base/07"],
        "dense_source_refs": _others(4) + [gold],
        "sparse_source_refs": _others(5, 10),
        "source_refs": _others(5),
    }
    result = stages_of(question)
    assert result["lost_at"] == "fusion"
    assert _by_name(result)["dense"]["rank"] == 5


def test_a_half_that_did_not_find_it_did_not_lose_it() -> None:
    gold = _ref(1, "07")
    question = {"expected_refs": ["base/07"], "dense_source_refs": [gold],
                "sparse_source_refs": _others(3), "source_refs": [gold]}
    result = stages_of(question)
    assert result["lost_at"] is None and result["found"] is True


def test_the_cut_to_the_context_is_a_stage_of_its_own() -> None:
    """Without a reranker, the window and the context differ by the cut, and
    a source past the cut is lost there."""
    gold = _ref(1, "07")
    question = {"expected_refs": ["base/07"],
                "candidate_source_refs": _others(6) + [gold], "source_refs": _others(5)}
    result = stages_of(question)
    assert [s["stage"] for s in result["stages"]] == ["retrieval", "context"]
    assert result["lost_at"] == "context"


def test_a_source_nobody_found_is_not_lost_anywhere() -> None:
    question = {"expected_refs": ["base/07"], "source_refs": _others(5)}
    result = stages_of(question)
    assert result["found"] is False and result["lost_at"] is None


def test_a_run_stored_before_the_halves_gets_what_it_recorded_and_no_more() -> None:
    gold = _ref(1, "07")
    result = stages_of({"expected_refs": ["base/07"], "source_refs": [gold]})
    assert [s["stage"] for s in result["stages"]] == ["retrieval", "context"]
    assert _by_name(result)["context"]["same_as"] == "retrieval"


def test_the_best_of_several_needed_sources_counts() -> None:
    question = {"expected_refs": ["base/07", "base/09"],
                "source_refs": _others(2) + [_ref(1, "09")] + [_ref(1, "07")]}
    assert _by_name(stages_of(question))["context"]["rank"] == 3


def test_a_question_about_the_whole_corpus_is_ranked_nowhere() -> None:
    """It names the documents it was drawn from and not every source that
    answers it, so a rank against them would be a claim about the list."""
    gold = _ref(1, "07")
    question = {"scope": "global", "expected_refs": ["base/07"],
                "dense_source_refs": [gold], "source_refs": [gold]}
    result = stages_of(question)
    assert all(s["rank"] is None for s in result["stages"])
    assert result["lost_at"] is None and result["found"] is None
    assert [s["size"] for s in result["stages"]] == [1, 1, 1]


def test_the_run_page_payload_carries_each_question_s_stages() -> None:
    """Attached where the funnel verdict is, so the page reads both from one
    payload and a stored run needs nothing recomputed by the browser."""
    from services.api_gateway.routers.experiments import _attach_stages

    gold = _ref(1, "07")
    questions = [{"expected_refs": ["base/07"], "dense_source_refs": [gold], "source_refs": []}]
    _attach_stages(questions)
    assert questions[0]["stages"]["lost_at"] == "fusion"
