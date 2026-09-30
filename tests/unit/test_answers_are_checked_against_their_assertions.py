"""An answer is checked against the statements its question says it must make.

The judge is a model, so everything here stands in for it: what is checked is
what the platform does with the judge's replies, and never whether a model
reads Russian well. That was measured separately, before any of this was
written, and the measurement is quoted in `eval/judge_model.py`.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.eval.assertions import build_prompt, coverage, language_of, parse_verdict
from core.eval.detectors import detect_assertions_not_judged
from core.models import Answer, SourceRef
from services.api_gateway.routers import experiments as E


class _Embedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0, 0.0] for t in texts]


class _Judge:
    """A model that answers from a script, and can be made to stop answering."""

    def __init__(self, replies: list[str], fail_after: int | None = None):
        self._replies = list(replies)
        self._fail_after = fail_after
        self.prompts: list[str] = []

    def generate(self, prompt: str, **params: Any) -> str:
        if self._fail_after is not None and len(self.prompts) >= self._fail_after:
            self.prompts.append(prompt)
            raise ConnectionError("connection refused")
        self.prompts.append(prompt)
        return self._replies.pop(0)


def _judge(replies: list[str], **kw: Any) -> tuple[E._AssertionJudge, _Judge]:
    model = _Judge(replies, **kw)
    return E._AssertionJudge(model, "ollama", "judge-model", "http://judge.test"), model


def _question(**extra: Any) -> dict:
    return {"question": "What is required?", "ground_truth": "r", "article_refs": ["SRC001/44"],
            "answerability": "answerable",
            "assertions": ["Calibration is done every three months.", "Only a lab verifies."],
            **extra}


def _answer(text: str = "Every three months; a lab verifies.", **metadata: Any) -> Answer:
    ref = SourceRef(doc_id="d1", chunk_id="c1", chunk_text="ctx", source_code="SRC001", article_no="44")
    return Answer(text=text, source_refs=[ref], metadata=metadata)


# ── The prompt and the reply ──────────────────────────────────────────────────

def test_the_prompt_follows_the_statement_s_language() -> None:
    assert language_of("Калибровка выполняется раз в три месяца.") == "ru"
    assert language_of("Calibration is done every three months.") == "en"
    assert "Утверждение:" in build_prompt("an answer in English", "Поверку выполняет лаборатория.")


def test_a_reply_nobody_can_read_is_no_verdict() -> None:
    assert parse_verdict('{"stated": true}') is True
    assert parse_verdict('{"stated": false}') is False
    assert parse_verdict('{"stated": "yes"}') is None
    assert parse_verdict("I think it is stated.") is None


def test_a_share_with_a_missing_verdict_is_not_a_share() -> None:
    assert coverage([{"stated": True}, {"stated": False}]) == 0.5
    assert coverage([{"stated": True}, {"stated": None}]) is None
    assert coverage([]) is None


# ── The judge ─────────────────────────────────────────────────────────────────

def test_one_statement_is_asked_per_call() -> None:
    judge, model = _judge(['{"stated": true}', '{"stated": false}'])
    verdicts = judge.verdicts("an answer", ["first", "second"])
    assert [v["stated"] for v in verdicts] == [True, False]
    assert len(model.prompts) == 2 and "first" in model.prompts[0] and "second" in model.prompts[1]


def test_after_the_judge_stops_answering_nothing_more_is_asked() -> None:
    """An unreachable server would otherwise cost one timeout per statement
    per question, for the whole run."""
    judge, model = _judge(['{"stated": true}'], fail_after=1)
    first = judge.verdicts("answer", ["a", "b", "c"])
    second = judge.verdicts("answer", ["d"])
    assert [v["stated"] for v in first + second] == [True, None, None, None]
    assert len(model.prompts) == 2
    assert judge.record()["gap"] == "transport_failed"
    assert "connection refused" in judge.record()["note"]


def test_a_prompt_the_server_cut_gives_no_verdict() -> None:
    class _Cut(_Judge):
        def prompt_was_cut(self) -> bool:
            return True

    judge = E._AssertionJudge(_Cut(['{"stated": true}']), "ollama", "m", "u")
    assert judge.verdicts("answer", ["a"])[0]["stated"] is None
    assert judge.record()["prompts_cut"] == 1


# ── The evaluator ─────────────────────────────────────────────────────────────

def test_the_share_of_statements_made_is_a_metric() -> None:
    judge, _ = _judge(['{"stated": true}', '{"stated": false}'])
    ev = E._CompositeEvaluator(embedder=_Embedder(), top_k=5, judge=judge)
    metrics, extras = ev.evaluate_in_full(_question(), _answer())
    assert metrics["assertion_coverage"] == 0.5
    assert [v["stated"] for v in extras["assertion_verdicts"]] == [True, False]


def test_an_unreadable_reply_leaves_the_question_without_the_metric() -> None:
    judge, _ = _judge(['{"stated": true}', "not json"])
    ev = E._CompositeEvaluator(embedder=_Embedder(), top_k=5, judge=judge)
    metrics, extras = ev.evaluate_in_full(_question(), _answer())
    assert "assertion_coverage" not in metrics
    assert [v["stated"] for v in extras["assertion_verdicts"]] == [True, None]


def test_without_a_judge_the_statements_are_recorded_as_unjudged() -> None:
    ev = E._CompositeEvaluator(embedder=_Embedder(), top_k=5, judge=None)
    metrics, extras = ev.evaluate_in_full(_question(), _answer())
    assert "assertion_coverage" not in metrics
    assert [v["stated"] for v in extras["assertion_verdicts"]] == [None, None]


@pytest.mark.parametrize("question, answer", [
    (_question(answerability="out_of_scope", article_refs=[]), _answer()),
    (_question(), _answer("", retrieval_only=True)),
    ({**_question(), "assertions": []}, _answer()),
])
def test_nobody_is_asked_where_there_is_nothing_to_judge(question: dict, answer: Answer) -> None:
    judge, model = _judge([])
    ev = E._CompositeEvaluator(embedder=_Embedder(), top_k=5, judge=judge)
    metrics, extras = ev.evaluate_in_full(question, answer)
    assert model.prompts == [] and extras["assertion_verdicts"] == []
    assert "assertion_coverage" not in metrics


# ── What the run says about it afterwards ────────────────────────────────────

def _run(questions: list[dict], judge: dict | None = None) -> dict:
    return {"config": {}, "judge": judge or {}, "question_results": questions}


def test_a_run_whose_judge_reached_only_some_questions_says_so() -> None:
    finding = detect_assertions_not_judged(_run([
        {"assertion_verdicts": [{"stated": True}], "metrics": {"assertion_coverage": 1.0}},
        {"assertion_verdicts": [{"stated": None}], "metrics": {}},
    ], judge={"gap": "transport_failed", "note": "ConnectionError: refused"}))
    assert finding is not None
    assert finding.params["unjudged"] == 1 and finding.params["carrying"] == 2
    assert finding.params["reason"] == "transport_failed"


def test_a_fully_judged_run_and_a_run_without_assertions_say_nothing() -> None:
    assert detect_assertions_not_judged(_run([
        {"assertion_verdicts": [{"stated": True}], "metrics": {"assertion_coverage": 1.0}}])) is None
    assert detect_assertions_not_judged(_run([{"metrics": {"retrieval_recall_at_k": 1.0}}])) is None


def test_two_runs_judged_by_different_models_are_flagged() -> None:
    from core.experiment.compare import check_comparability
    from core.experiment.config import ComponentRef, ExperimentConfig
    from core.experiment.runner import ExperimentResult

    def run(model: str) -> ExperimentResult:
        r = ExperimentResult(config=ExperimentConfig(
            name="r", chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
            embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
            generator=ComponentRef(kind="generator", component_id="ollama")))
        r.aggregate_metrics = {"assertion_coverage": 0.5}
        r.judge = {"model": model}
        return r

    ids = lambda a, b: {w.id for w in check_comparability(run(a), run(b))}  # noqa: E731
    assert "different_judge" in ids("qwen3:32b", "qwen2.5:7b")
    assert "different_judge" not in ids("qwen3:32b", "qwen3:32b")


# ── The generator's window ────────────────────────────────────────────────────

def test_the_window_is_sent_only_when_named_and_a_cut_prompt_is_recognised(monkeypatch) -> None:
    import adapters.ollama_generator as og

    sent: list[dict] = []

    class _Response:
        def __init__(self, count: int):
            self._count = count

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"response": '{"stated": true}', "prompt_eval_count": self._count, "eval_count": 3}

    counts = iter([2050, 700, 900])
    monkeypatch.setattr(og.httpx, "post",
                        lambda url, json, timeout: (sent.append(json), _Response(next(counts)))[1])

    named = og.OllamaGenerator(base_url="http://x", model="m", num_ctx=4096)
    named.generate("p")
    assert sent[-1]["options"]["num_ctx"] == 4096
    assert named.prompt_was_cut() is True
    named.generate("p")
    assert named.prompt_was_cut() is False

    unnamed = og.OllamaGenerator(base_url="http://x", model="m")
    unnamed.generate("p")
    assert "num_ctx" not in sent[-1]["options"]
    assert unnamed.prompt_was_cut() is False
    assert named.with_model("other")._num_ctx == 4096


# ── Finding the judge ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("tags, gap", [
    ({"models": [{"name": "qwen2.5:7b"}]}, "model_not_pulled"),
    (ConnectionError("refused"), "unreachable"),
])
def test_a_judge_that_cannot_be_had_is_named_with_why(monkeypatch, tags, gap) -> None:
    import httpx

    import services.api_gateway.routers.corpus as corpus
    import services.api_gateway.routers.settings as settings

    async def no_settings(realm_id=None):
        return {}

    async def no_resource(realm_id, kind):
        return None

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url):
            if isinstance(tags, Exception):
                raise tags

            class _R:
                def json(self_inner):
                    return tags
            return _R()

    monkeypatch.setattr(settings, "_get_settings_doc", no_settings)
    monkeypatch.setattr(corpus, "_get_realm_resource", no_resource)
    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    judge = asyncio.run(E._resolve_judge("demo"))
    assert isinstance(judge, E._UnavailableJudge)
    assert judge.record()["gap"] == gap
    assert [v["stated"] for v in judge.verdicts("a", ["x"])] == [None]


# ── Judging after the run ─────────────────────────────────────────────────────

def test_a_deferred_judge_asks_nothing_until_every_answer_exists() -> None:
    """Measured: the judge alone answered in under a second a statement, and
    in nine to ten when the generator had run just before it, because the
    server keeps one model loaded and reloads it on each switch."""
    from core.experiment.config import ComponentRef, ExperimentConfig
    from core.experiment.runner import ExperimentResult, QuestionResult

    judge, model = _judge(['{"stated": true}', '{"stated": false}', '{"stated": true}', '{"stated": true}'])
    ev = E._CompositeEvaluator(embedder=_Embedder(), top_k=5, judge=judge, defer_judging=True)
    result = ExperimentResult(config=ExperimentConfig(
        name="r", chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
        embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
        generator=ComponentRef(kind="generator", component_id="ollama")))
    for qid in ("q1", "q2"):
        metrics, extras = ev.evaluate_in_full(_question(id=qid), _answer())
        assert "assertion_coverage" not in metrics
        result.question_results.append(QuestionResult(
            question_id=qid, question="?", reference_answer="r", generated_answer="a",
            metrics=metrics, assertion_verdicts=extras["assertion_verdicts"]))
    assert model.prompts == []

    ev.judge_pending(result)
    assert len(model.prompts) == 4
    assert [qr.metrics["assertion_coverage"] for qr in result.question_results] == [0.5, 1.0]
    assert result.aggregate_metrics["assertion_coverage"] == 0.75
    assert [v["stated"] for v in result.question_results[0].assertion_verdicts] == [True, False]
