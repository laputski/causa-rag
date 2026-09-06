"""A RAG server that answers badly on purpose, and stays apart from the honest one.

The reference server exists so the external-RAG contract is checked by an
honest implementation of it. Putting code into that server which lies on
purpose would destroy exactly what it is for, so the lying lives outside it and
is applied to the answer the honest pipeline produced.

Each mode is named for the catalogue entry it stages, and every assertion below
is about what a caller receives, which is all the platform can see of somebody
else's system.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from services.faulty_rag_server import main as faulty
from services.reference_rag_server.main import ExternalRagRequest

ROOT = Path(__file__).resolve().parents[2]


def _response(answer: str = "Calibration is every three months (Fragment 1).",
              sources: int = 3) -> dict[str, Any]:
    return {
        "answer": answer,
        "sources": [
            {"doc_id": f"base-ru/{n:02d}", "chunk_id": f"c{n}",
             "chunk_text": f"Section {n}. " + "text that runs on for a while " * 4,
             "score": 1.0 - n / 10}
            for n in range(1, sources + 1)
        ],
    }


def _apply(name: str, response: dict[str, Any] | None = None) -> dict[str, Any]:
    fault = next(f for f in faulty.FAULTS if f.name == name)
    original = response or _response()
    before = {**original, "sources": [dict(s) for s in original["sources"]]}
    out = fault.apply(original, ExternalRagRequest(query="q"))
    assert original == before, f"{name} changed the response it was given"
    return out


# ── the modes ─────────────────────────────────────────────────────────────────

def test_the_control_changes_nothing() -> None:
    """Every pair needs a half that behaves, and it has to be the same server:
    a control run against a different program compares two programs."""
    original = _response()
    assert _apply("none", original) == original


def test_never_refusing_answers_what_the_corpus_does_not_cover() -> None:
    refused = _response(answer=faulty.REFUSAL)
    assert _apply("F19_never_refuses", refused)["answer"] != faulty.REFUSAL
    assert _apply("F19_never_refuses", refused)["answer"].strip()


def test_never_refusing_leaves_a_real_answer_alone() -> None:
    """The mode is about the questions with no answer. Rewriting the others
    would stage a second failure and make the pair prove neither."""
    answered = _response()
    assert _apply("F19_never_refuses", answered)["answer"] == answered["answer"]


def test_always_refusing_refuses_what_the_corpus_answers() -> None:
    assert _apply("F32_always_refuses")["answer"] == faulty.REFUSAL


def test_returning_nothing_leaves_the_sources_in_place() -> None:
    """Retrieval succeeded and generation produced nothing, which is what a
    reasoning model does when it spends its budget thinking. Dropping the
    sources too would look like a retrieval failure instead."""
    out = _apply("F28_returns_nothing")
    assert out["answer"] == ""
    assert len(out["sources"]) == 3


def test_the_wrong_citation_moves_the_number_and_keeps_the_sentence() -> None:
    out = _apply("F29_cites_the_wrong_fragment")
    assert "(Fragment 2)" in out["answer"]
    assert "(Fragment 1)" not in out["answer"]
    assert "Calibration is every three months" in out["answer"]


def test_the_wrong_citation_leaves_a_single_source_alone() -> None:
    """With one source there is no other number to move to."""
    single = _response(sources=1)
    assert _apply("F29_cites_the_wrong_fragment", single) == single


def test_the_wrong_citation_survives_an_answer_with_no_sources() -> None:
    """A refusal carries none, and dividing by their count would raise inside
    a server whose whole job is to answer badly and keep answering."""
    empty = {"answer": "Calibration is every three months (Fragment 1).", "sources": []}
    assert _apply("F29_cites_the_wrong_fragment", empty) == empty


def test_ignoring_the_context_keeps_the_sources_it_ignores() -> None:
    """The failure is invisible to a retrieval metric precisely because recall
    stays at one: the right sources are there and the answer is not from them."""
    out = _apply("F31_ignores_the_context")
    assert "three months" not in out["answer"]
    assert len(out["sources"]) == 3


def test_burying_the_middle_answers_from_the_edges() -> None:
    out = _apply("F30_buries_the_middle")
    assert "(Fragment 1)" in out["answer"]
    assert "(Fragment 3)" in out["answer"]
    assert len(out["sources"]) == 3


def test_cutting_sources_leaves_them_starting_mid_thought() -> None:
    out = _apply("F05_cuts_sources_mid_sentence")
    for source in out["sources"]:
        assert not source["chunk_text"].startswith("Section")
        assert source["chunk_text"]


# ── the modes as a set ────────────────────────────────────────────────────────

def test_every_mode_but_the_control_names_a_catalogue_entry_that_exists() -> None:
    from core.eval.atlas import FAILURES

    known = {f.id for f in FAILURES}
    for fault in faulty.FAULTS:
        if fault.name == "none":
            assert fault.provokes == (), "the control stages nothing, by definition"
            continue
        assert fault.provokes, f"{fault.name} stages nothing, so nothing needs it"
        assert set(fault.provokes) <= known, f"{fault.name} names {fault.provokes}"


def test_every_entry_it_names_is_one_this_instrument_is_meant_to_stage() -> None:
    """A mode pointing at a corpus defect would stage nothing here, and its
    pair would be measuring the wrong instrument."""
    from core.eval.atlas import FAILURES

    by_id = {f.id: f for f in FAILURES}
    for fault in faulty.FAULTS:
        for failure_id in fault.provokes:
            assert by_id[failure_id].instrument == "faulty_rag", (
                f"{failure_id} is staged by {by_id[failure_id].instrument!r}"
            )


def test_every_entry_of_this_instrument_has_a_mode() -> None:
    """The other direction: an entry the catalogue assigns to this instrument
    and nothing here stages is a gap that would otherwise stay quiet."""
    from core.eval.atlas import FAILURES

    assigned = {f.id for f in FAILURES if f.instrument == "faulty_rag"}
    staged = {i for fault in faulty.FAULTS for i in fault.provokes}
    assert assigned <= staged, f"entries with no mode: {sorted(assigned - staged)}"


def test_a_misspelled_mode_stops_the_server() -> None:
    """A server started with a mode nobody defined would answer honestly while
    its operator believed it was staging a failure, and every pair built on it
    would prove the opposite of what it claimed."""
    import os
    from unittest.mock import patch

    with patch.dict(os.environ, {"RAG_FAULT": "F19_never_refuse"}), pytest.raises(SystemExit) as raised:
        faulty.active_fault()
    assert "Unknown RAG_FAULT" in str(raised.value)
    assert "F19_never_refuses" in str(raised.value), "it does not say what the names are"


def test_saying_nothing_gives_the_control() -> None:
    import os
    from unittest.mock import patch

    with patch.dict(os.environ, {}, clear=True):
        assert faulty.active_fault().name == "none"


# ── the two servers stay apart ────────────────────────────────────────────────

def test_the_honest_server_has_never_heard_of_this_one() -> None:
    """The measure that matters most, and the one that erodes by editing.

    The reference server exists so the contract is checked by an honest
    implementation. A branch inside it that lies on request would leave nothing
    checking the contract at all.
    """
    honest = (ROOT / "services" / "reference_rag_server" / "main.py").read_text(encoding="utf-8")
    for word in ("faulty", "RAG_FAULT", "distort"):
        assert word not in honest, f"the reference server mentions {word!r}"


def test_this_one_reuses_the_contract_instead_of_copying_it() -> None:
    """A second copy of the request model or the pipeline resolution would
    drift, and then this server would be testing a contract nobody serves."""
    source = (ROOT / "services" / "faulty_rag_server" / "main.py").read_text(encoding="utf-8")
    assert "from services.reference_rag_server.main import" in source
    assert "_resolve_pipeline" in source


def test_the_two_listen_on_different_ports() -> None:
    """So that reaching the wrong one is a different command and not a typo."""
    from services.reference_rag_server.main import PORT as honest_port

    assert honest_port != faulty.PORT
    assert honest_port == faulty.REFERENCE_PORT


def test_both_can_be_started_and_the_commands_say_which_is_which() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "reference-rag:" in makefile
    assert "faulty-rag:" in makefile
    assert "on purpose" in makefile


def test_the_realm_record_points_at_the_port_the_server_listens_on() -> None:
    """The number is written twice: importing the server into the seed would
    pull the whole gateway into it. A record pointing at a free port fails as
    a connection refused, minutes into a run, and reads as the server being
    down, and never as the record being wrong."""
    from tools.seed_proving_ground import FAULTY_RAG_PORT, build_bundle

    assert FAULTY_RAG_PORT == faulty.PORT
    record = build_bundle()["external_rags"][0]
    assert record["url"] == f"http://localhost:{faulty.PORT}/"
    assert record["retrieve_endpoint"] == f"http://localhost:{faulty.PORT}/retrieve"


def test_the_realm_record_says_what_it_is_registering() -> None:
    """Somebody meeting this record in a list of external systems has to be
    able to tell it apart from a real one at a glance."""
    from tools.seed_proving_ground import build_bundle

    record = build_bundle()["external_rags"][0]
    assert "on purpose" in record["description"]
    assert "make faulty-rag" in record["description"]


def test_it_is_registered_in_the_proving_ground_and_nowhere_else() -> None:
    """A record like this in the demo realm would put a system that lies into
    the example the platform ships."""
    import json
    from pathlib import Path

    demo = json.loads((ROOT / "ui" / "public" / "demo.realm.json").read_text(encoding="utf-8"))
    assert demo.get("external_rags") == [], "the demo realm registers an external system"
    assert Path(ROOT / "tools" / "seed_proving_ground.py").read_text(encoding="utf-8").count(
        "faulty-rag") >= 1
