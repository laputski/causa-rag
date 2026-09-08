"""What reading a run costs, and what reading a run is allowed to touch.

The plan this proving ground was built from names this file twice and lists it
as one of the commands that check the work. It did not exist. `pytest` on a
path that matches nothing exits 4 and prints "no tests ran", which reads like
success to anyone glancing at the output and to any `&&` chain, so the absence
of the guard was itself invisible for as long as the guard was.

Two things are held here, and they are the two properties the whole design of
the detectors rests on.

**Cost.** Every detector runs when somebody opens a run, not when the run is
made, and that is why adding one is cheap and why a detector written today
applies to every run stored yesterday. It only stays cheap while the cost of
reading stays small, and the cost is linear in the total length of the
answers, because a regular expression over the answer text is what dominates
it. Measured when the plan was written: 0.081 ms per thousand characters over
six loads spanning an eightfold range. Measured here with twenty findings
where there were twelve then: 0.079 to 0.080 over five trials.

The answers below are prose and never one character repeated, and the
difference is not cosmetic. Measured on a string of one letter the same load
reads at 0.095, because what the pattern engine does with a run of one
character is not what it does with words, so a payload chosen for convenience
would have reported a rise that is not there.

The thresholds are set from that spread and from what a planted defect costs,
so each says what it catches. Five clean trials sat between 0.079 and 0.080
and put the ratio below at 1.00 to 1.02; a detector doing a small fixed amount
of work per question, planted to check, read at 0.124 and 1.62.

**Purity.** The same property from the other side. A detector that opens a
file or a socket has left the class it was designed into: it would make
opening a run wait on a service, and it would make the cost above depend on
something no measurement here can see. So the whole set is run with the
filesystem and the network taken away, on a real stored run.
"""
from __future__ import annotations

import builtins
import json
import pathlib
import socket
import time
from typing import Any

import pytest

from core.eval.detectors import run_detectors

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Milliseconds per thousand characters of answer text. A ratchet: it may fall
#: and may never rise. Two and a half times the measured cost, which is room
#: for a slower machine and not for another detector.
BUDGET_MS_PER_1000_CHARS = 0.2

#: The load the plan specifies, so a number measured now can be compared with
#: the number measured then.
QUESTIONS = 200
CHARACTERS = 1000


def _synthetic(questions: int, characters: int) -> dict[str, Any]:
    """A run of a given size, with nothing in it that any detector reddens.

    The answers are prose and not one repeated character, because the pattern
    that dominates the cost is a search for the phrasings of a refusal and a
    string of one letter is not what it walks over in a real run.
    """
    sentence = ("Порядок проверки узла описан в разделе выше и повторяется "
                "без изменений на каждом приборе. ")
    answer = (sentence * (characters // len(sentence) + 1))[:characters]
    return {
        "config": {"pipeline_source": "in_process"},
        "aggregate_metrics": {},
        "question_results": [
            {"question": f"question {i}", "generated_answer": answer,
             "metrics": {}, "source_refs": []}
            for i in range(questions)
        ],
    }


def test_reading_a_run_stays_inside_its_budget() -> None:
    run = _synthetic(QUESTIONS, CHARACTERS)
    run_detectors(run)  # once first, so an import is not counted as cost

    started = time.perf_counter()
    for _ in range(3):
        run_detectors(run)
    milliseconds = (time.perf_counter() - started) * 1000 / 3

    thousands = QUESTIONS * CHARACTERS / 1000
    rate = milliseconds / thousands
    assert rate < BUDGET_MS_PER_1000_CHARS, (
        f"reading a run of {QUESTIONS} answers of {CHARACTERS} characters costs "
        f"{milliseconds:.1f} ms, which is {rate:.3f} ms per thousand characters against a "
        f"budget of {BUDGET_MS_PER_1000_CHARS}. Something in the read path now does more than "
        "walk the text it was given."
    )


def test_the_cost_follows_the_length_of_the_answers_and_not_the_questions() -> None:
    """The shape of the law, which is what makes the budget above meaningful.

    Two loads carrying the same total number of characters in a different
    number of questions cost the same. If that stops holding, the cost has
    moved to something per question and the budget is measuring the wrong
    thing.
    """
    def cost(questions: int, characters: int) -> float:
        run = _synthetic(questions, characters)
        run_detectors(run)
        started = time.perf_counter()
        for _ in range(3):
            run_detectors(run)
        return (time.perf_counter() - started) * 1000 / 3

    few_long = cost(100, 2000)
    many_short = cost(400, 500)
    assert few_long > 0 and many_short > 0
    ratio = max(few_long, many_short) / min(few_long, many_short)
    # Clean trials sit at 1.00 to 1.02 and a small fixed cost per question
    # puts it at 1.62, so this catches per-question work worth about half a
    # read and says nothing about anything smaller.
    assert ratio < 1.3, (
        f"the same total length costs {few_long:.1f} ms in a hundred questions and "
        f"{many_short:.1f} ms in four hundred, a ratio of {ratio:.1f}. The cost has moved to "
        "something counted per question."
    )


def test_reading_a_run_touches_no_file_and_no_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run on a real stored run, and never on a payload written here: a
    fixture exercises the branches somebody thought of, and what this is
    looking for is a detector that reaches for something on a branch nobody
    did."""
    stored = sorted((ROOT / "eval" / "results" / "runs").glob("*.json"))
    if not stored:
        pytest.skip("NOT RUN: no stored run to read")

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the read path opened something")

    for run in stored[:5]:
        payload = json.loads(run.read_text(encoding="utf-8"))
        with monkeypatch.context() as taken_away:
            taken_away.setattr(builtins, "open", refuse)
            taken_away.setattr(pathlib.Path, "open", refuse)
            taken_away.setattr(pathlib.Path, "read_text", refuse)
            taken_away.setattr(pathlib.Path, "read_bytes", refuse)
            taken_away.setattr(socket, "socket", refuse)
            run_detectors(payload)


def test_the_purity_check_would_notice(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bait for the test above. A guard that takes the filesystem away and
    never checks that taking it away is felt is a guard that would pass on a
    monkeypatch that stopped working."""
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the read path opened something")

    with monkeypatch.context() as taken_away:
        taken_away.setattr(pathlib.Path, "read_text", refuse)
        with pytest.raises(AssertionError, match="opened something"):
            (ROOT / "pyproject.toml").read_text(encoding="utf-8")
