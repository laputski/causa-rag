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


def _response(answer: str = "Calibration is every three months, per section 2.",
              sources: int = 3) -> dict[str, Any]:
    # structural_path carries the bracketed label the chunker writes, because
    # that is what the platform is handed and what the citation signal reads.
    return {
        "answer": answer,
        "sources": [
            {"doc_id": f"base-ru/{n:02d}", "chunk_id": f"c{n}",
             "chunk_text": f"Section {n}. " + "text that runs on for a while " * 4,
             "structural_path": f"document/section[{n} Section {n}]",
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
    assert "section 3" in out["answer"]
    assert "section 2" not in out["answer"]
    assert "Calibration is every three months" in out["answer"]


def test_the_wrong_citation_moves_what_the_signal_actually_reads() -> None:
    """The half of this pair that rots is the mode going inert.

    Its first version moved a "(Fragment N)" marker, which the generator
    never writes and the signal never reads, so the broken run scored
    exactly like its control and the pair proved nothing. Asserting against
    the signal itself is what makes that failure visible here instead of
    two minutes into a live run.
    """
    from core.citation import citation_number_coverage
    from core.eval.retrieval_metrics import extract_ref_id
    from core.models import SourceRef

    original = _response()
    refs = [SourceRef(doc_id=s["doc_id"], chunk_id=s["chunk_id"], chunk_text=s["chunk_text"],
                      score=s["score"], structural_path=s["structural_path"])
            for s in original["sources"]]
    # Asked for, not written out: a hand-built ref id that the platform would
    # never produce makes the metric return None and the assertion pass for
    # a reason having nothing to do with the mode.
    article_refs = [extract_ref_id(refs[1].model_dump())]

    honest = citation_number_coverage(original["answer"], refs, article_refs)
    broken = citation_number_coverage(
        _apply("F29_cites_the_wrong_fragment", original)["answer"], refs, article_refs)
    assert honest == 1.0, honest
    assert broken == 0.0, broken


def test_the_wrong_citation_leaves_numbers_belonging_to_no_fragment_alone() -> None:
    """A year, a torque, a temperature: moving those would stage a second
    failure, and the entry is about the citation and nothing else."""
    out = _apply("F29_cites_the_wrong_fragment",
                 _response(answer="Tighten to 40 Nm, per section 2."))
    assert "40 Nm" in out["answer"]


def test_the_wrong_citation_leaves_a_single_source_alone() -> None:
    """With one source there is no other number to move to."""
    single = _response(sources=1)
    assert _apply("F29_cites_the_wrong_fragment", single) == single


def test_the_wrong_citation_survives_an_answer_with_no_sources() -> None:
    """A refusal carries none, and dividing by their count would raise inside
    a server whose whole job is to answer badly and keep answering."""
    empty = {"answer": "Calibration is every three months, per section 2.", "sources": []}
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


def test_cutting_sources_leaves_them_ending_mid_thought() -> None:
    out = _apply("F05_cuts_sources_mid_sentence")
    for before, after in zip(_response()["sources"], out["sources"], strict=True):
        assert after["chunk_text"]
        assert len(after["chunk_text"]) < len(before["chunk_text"])
        assert not after["chunk_text"].rstrip().endswith(".")


def test_cutting_sources_does_not_make_two_fragments_identical() -> None:
    """The mode stages one failure and has to stage only one.

    Its first version kept a fixed window out of the middle of each
    fragment. The corpus this runs against carries its distinctness in the
    opening, since every service card names its model there and then says the
    same things about it, so dropping the head left fragments byte-identical
    and a live run reported duplicates, which is a different entry, about a
    corpus the mode had not touched. The fixture below has that shape.
    """
    original = _response()
    for n, source in enumerate(original["sources"], start=1):
        source["chunk_text"] = (f"Model KL-{n}00. " + "The rest of this card is the same "
                                "sentence every card of this kind carries. ")
    texts = [s["chunk_text"] for s in _apply("F05_cuts_sources_mid_sentence", original)["sources"]]
    assert len(set(texts)) == len(texts), texts


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


def test_the_refusal_it_recognises_is_the_one_the_platform_counts() -> None:
    """The mode that answers everything has to recognise the refusals the
    platform will later count as refusals.

    Written first against two phrases guessed here, on the reasoning that a
    server playing the part of somebody else's system should not read this
    platform's internals. Measured against the generator's real output, that
    guess matched nothing: the mode replaced no answer, ran against nineteen
    questions, and scored exactly what the honest mode scored. A distortion
    that runs and provokes nothing is the failure this whole apparatus exists
    to stop.
    """
    from core.eval.detectors import NOT_FOUND_RE

    # Observed from the generator this proving ground runs, and not invented.
    real = "В предоставленных документах информация по данному вопросу отсутствует."
    assert NOT_FOUND_RE.search(real), "the platform no longer reads this as a refusal"
    assert faulty._reads_as_refusal(real)
    assert faulty._reads_as_refusal(faulty.REFUSAL)
    assert faulty._reads_as_refusal("")
    assert not faulty._reads_as_refusal("Калибровка выполняется раз в три месяца.")


def test_never_refusing_replaces_the_generator_s_own_refusal() -> None:
    """The pair of the test above, at the level of the mode."""
    real = _response(answer="В предоставленных документах информация по данному вопросу отсутствует.")
    out = _apply("F19_never_refuses", real)
    assert out["answer"] != real["answer"]
    assert not faulty._reads_as_refusal(out["answer"])
