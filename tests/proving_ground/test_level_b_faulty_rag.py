"""Paired baits for the failures that belong to whoever generates the answer.

A system that never refuses, that returns nothing, that cites the wrong
fragment, that answers from its own knowledge: none of those live in the
documents, the settings or the index, and the platform sees such a system only
through the external-RAG contract. `services/faulty_rag_server` is one, wrong
deliberately in one named way at a time.

Each pair starts that server twice, once in the control mode and once in the
mode under test, and runs the same questions through the platform against it.
Restarting is the price of the design: a mode that could change between two
requests of one run would describe no system anybody operates, so the mode
belongs to the process.

These runs generate. Every other suite here stops before the generator, and
these cannot: the failures are in the answer.
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.proving_ground.conftest import REALM, record

pytestmark = pytest.mark.proving_ground

ROOT = Path(__file__).resolve().parents[2]
CORPUS = "base-ru"
URL = "http://localhost:8092/"


def _up(timeout: float = 90.0) -> dict[str, Any]:
    import json

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(URL + "health", timeout=2) as answer:
                return json.loads(answer.read())
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(1)
    raise TimeoutError("the faulty server did not come up")


@contextlib.contextmanager
def running(fault: str) -> Iterator[dict[str, Any]]:
    """The server, in one mode, for the length of one run."""
    process = subprocess.Popen(
        ["python3", "-m", "services.faulty_rag_server.main"],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env={**os.environ, "RAG_FAULT": fault, "RAG_REALM": REALM, "USE_REAL_BGE_M3": "true"},
    )
    try:
        health = _up()
        assert health["fault"] == fault, (
            f"asked for {fault!r} and the server reports {health['fault']!r}; another one is "
            "probably still listening on that port"
        )
        yield health
    finally:
        process.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=20)


def _run(embedder: Any) -> dict[str, Any]:
    """The golden set through the platform, against whatever is on that port."""
    from core.experiment.config import ComponentRef, ExperimentConfig
    from core.experiment.runner import ExperimentRunner
    from core.registry import ComponentRegistry
    from eval.dataset import EvalDataset
    from services.api_gateway.routers.experiments import _CompositeEvaluator

    config = ExperimentConfig(
        name="faulty-rag", pipeline_source="http", http_endpoint=URL,
        chunking_strategy=ComponentRef(kind="chunker", component_id="structure_aware"),
        embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
        generator=ComponentRef(kind="generator", component_id="external"),
        pipeline_id="hybrid_rrf", corpus_id=CORPUS, top_k=5,
    )
    dataset = EvalDataset.from_jsonl(ROOT / "eval" / "golden" / f"{CORPUS}.v1.fast.jsonl")
    result = ExperimentRunner(ComponentRegistry()).run(
        config, dataset, realm_id=REALM,
        evaluator=_CompositeEvaluator(embedder, top_k=config.top_k),
    )
    return result.to_dict()


@pytest.fixture(scope="module")
def control(embedder: Any) -> dict[str, Any]:
    """The honest half, run once. The same server, the same corpus, the same
    questions: only the mode differs from every half below."""
    with running("none"):
        return _run(embedder)


def _signals(run: dict[str, Any]) -> set[str]:
    from core.eval.detectors import run_detectors
    return {f"detector:{item.id}" for item in run_detectors(run)}


def _details(run: dict[str, Any], only: set[str]) -> dict[str, str]:
    """What the signals that spoke actually said. A reverse bait that fails
    has to say what named the failure, or the next reader repeats the run."""
    from core.eval.detectors import run_detectors
    return {f"detector:{item.id}": item.detail
            for item in run_detectors(run) if f"detector:{item.id}" in only}


def _metric(run: dict[str, Any], name: str) -> float:
    return float(run["aggregate_metrics"].get(name, -1.0))


def test_the_control_answers_and_is_not_reported_as_broken(control: dict[str, Any]) -> None:
    """The premise of every pair below. A control that already looks broken
    would make each of them compare two broken halves."""
    answered = [q for q in control["question_results"] if (q.get("generated_answer") or "").strip()]
    assert len(answered) > len(control["question_results"]) // 2, (
        f"the honest mode answered {len(answered)} of {len(control['question_results'])}"
    )
    assert "detector:empty_answers" not in _signals(control)


def test_F28_a_server_that_returns_nothing_is_reported(
    embedder: Any, control: dict[str, Any]
) -> None:
    """Retrieval succeeded and the answer is empty, which is what a reasoning
    model does when it spends its whole budget thinking."""
    with running("F28_returns_nothing"):
        broken = _run(embedder)
    assert "detector:empty_answers" in _signals(broken), (
        f"every answer was empty and nothing said so: {sorted(_signals(broken))}"
    )
    record("F28", "an external system that returns no answer while retrieval succeeded",
           signals=sorted(_signals(broken)), signals_on_the_control=sorted(_signals(control)))


def test_F32_a_server_that_refuses_everything_is_reported(
    embedder: Any, control: dict[str, Any]
) -> None:
    """Refusal miscalibrated in one direction: a threshold so high that
    nothing clears it, including what the corpus answers plainly."""
    with running("F32_always_refuses"):
        broken = _run(embedder)
    assert _metric(broken, "correct_refusal") < _metric(control, "correct_refusal"), (
        f"refusing everything scored no worse than answering: "
        f"{_metric(broken, 'correct_refusal')} against {_metric(control, 'correct_refusal')}"
    )
    assert "detector:incorrect_refusals" in _signals(broken), (
        f"every answerable question was refused and nothing said so: {sorted(_signals(broken))}"
    )
    record("F32", "an external system that refuses what its corpus answers plainly",
           correct_refusal_control=_metric(control, "correct_refusal"),
           correct_refusal_broken=_metric(broken, "correct_refusal"),
           signals=sorted(_signals(broken)))


def test_F19_a_server_that_never_refuses_is_measured(
    embedder: Any, control: dict[str, Any]
) -> None:
    """The other direction, and the one a retrieval metric cannot see: every
    question is answered, including the two the corpus does not cover."""
    with running("F19_never_refuses"):
        broken = _run(embedder)
    assert _metric(broken, "correct_refusal") < _metric(control, "correct_refusal"), (
        f"answering the unanswerable scored no worse: {_metric(broken, 'correct_refusal')} "
        f"against {_metric(control, 'correct_refusal')}"
    )
    record("F19", "an external system with no confidence threshold answers what it cannot",
           correct_refusal_control=_metric(control, "correct_refusal"),
           correct_refusal_broken=_metric(broken, "correct_refusal"))


def test_F31_a_server_that_ignores_its_own_sources_is_measured(
    embedder: Any, control: dict[str, Any]
) -> None:
    """The failure a retrieval metric is blind to by construction: recall stays
    where it was, because the right sources are returned, and the answer does
    not come from them."""
    with running("F31_ignores_the_context"):
        broken = _run(embedder)
    assert _metric(broken, "retrieval_recall_at_k") == _metric(control, "retrieval_recall_at_k"), (
        "retrieval moved, so this pair changed two things"
    )
    assert _metric(broken, "context_support") < _metric(control, "context_support"), (
        f"an answer from nowhere scored as well as one from the sources: "
        f"{_metric(broken, 'context_support')} against {_metric(control, 'context_support')}"
    )
    appeared = _signals(broken) - _signals(control)
    record("F31", "an external system that answers from its own knowledge, sources unused",
           recall_unchanged=_metric(broken, "retrieval_recall_at_k"),
           context_support_control=_metric(control, "context_support"),
           context_support_broken=_metric(broken, "context_support"),
           signals_that_appeared=sorted(appeared), what_they_said=_details(broken, appeared))


def test_F05_sources_cut_mid_sentence_go_unremarked(
    embedder: Any, control: dict[str, Any]
) -> None:
    """A reverse bait: the failure is staged and nothing speaks.

    The catalogue says nothing measures whether a fragment boundary falls
    inside a sentence, and a claim of not being caught is a claim like any
    other. This is its evidence: the fragments come back cut at a fixed
    width, and no detector says a word about it.
    """
    with running("F05_cuts_sources_mid_sentence"):
        broken = _run(embedder)
    appeared = _signals(broken) - _signals(control)
    assert not appeared, (
        f"something did name it after all, and the entry says nothing does: "
        f"{sorted(appeared)}: {_details(broken, appeared)}"
    )
    record("F05", "fragments cut at a fixed width, and no signal names the cut",
           signals=sorted(_signals(broken)), signals_on_the_control=sorted(_signals(control)),
           context_support_control=_metric(control, "context_support"),
           context_support_broken=_metric(broken, "context_support"))


def test_F30_an_overfilled_context_answered_from_its_edges_goes_unremarked(
    embedder: Any, control: dict[str, Any]
) -> None:
    """The second reverse bait, and the same shape.

    Position bias is computed on demand and is not surfaced as a signal, so
    an answer drawn from the first source and the last, with everything
    between them returned and unread, passes without a word.
    """
    with running("F30_buries_the_middle"):
        broken = _run(embedder)
    appeared = _signals(broken) - _signals(control)
    said = _details(broken, appeared)
    assert not (appeared - {"detector:layer_bottleneck"}), (
        f"something named it that the entry did not expect: {sorted(appeared)}"
    )
    record("F30", "a context filled past what was asked for, answered from its edges",
           signals_that_appeared=sorted(appeared), what_they_said=said,
           signals_on_the_control=sorted(_signals(control)),
           context_support_control=_metric(control, "context_support"),
           context_support_broken=_metric(broken, "context_support"))


def test_F29_is_not_stageable_on_this_corpus_and_the_reason_is_measured(
    embedder: Any, control: dict[str, Any]
) -> None:
    """The entry's signal has nothing to read here, and that is a fact about
    the corpus and not about the instrument.

    `citation_number_coverage` looks for the structural number of a retrieved
    fragment occurring in the answer text. In this corpus only the top-level
    heading of each document is numbered; every subsection is a bare title,
    and retrieval returns subsections. So the signal has no candidate number
    for almost every question, and a run against a server that moves every
    citation scores exactly like its control, which was measured twice before
    it was traced to this.
    """
    import re

    labelled = numbered = 0
    for question in control["question_results"]:
        for source in question.get("source_refs") or []:
            label = re.search(r"\[([^\]]+)\]", source.get("structural_path") or "")
            if not label:
                continue
            labelled += 1
            if re.search(r"\d", label.group(1)):
                numbered += 1

    assert labelled > 0, "no retrieved fragment carries a bracketed label at all"
    assert numbered < labelled // 2, (
        f"{numbered} of {labelled} retrieved labels carry a number, so the signal has "
        "candidates after all and this entry is stageable here"
    )
    record("F29", "not staged: retrieval returns subsections, whose labels carry no number, "
                  "so the signal the entry names has nothing to look for",
           reproduced=False,
           retrieved_labels=labelled, of_them_numbered=numbered,
           coverage_recorded_by_the_control=control["aggregate_metrics"].get(
               "citation_number_coverage"),
           baited_at="tests/unit/test_atlas_baits.py, on a payload whose labels carry numbers")
