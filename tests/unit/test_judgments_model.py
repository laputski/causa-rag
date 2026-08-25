"""core/judgments.

Pins the properties the rework rests on: a judgment records an observation
rather than an instruction, it survives a round trip through the file, and it
converts into the two things that make it more useful than the retrieval pin
it replaces — a permanent test case and a preference pair.
"""
from __future__ import annotations

from pathlib import Path

from core.judgments import (
    JudgedChunk,
    RelevanceJudgment,
    judgment_from_dict,
    judgment_to_dict,
    preference_pairs,
    to_golden_question,
)
from core.judgments.store import (
    append_judgment,
    judgments_path,
    load_judgments,
    rewrite_judgments,
)


def _judgment(**overrides) -> RelevanceJudgment:
    base = dict(
        id="j1",
        realm_id="acme",
        corpus_id="handbook_01",
        question="What is the appeal deadline?",
        relevant=(JudgedChunk(chunk_id="c1", ref_id="SRC001/44"),),
        irrelevant=(JudgedChunk(chunk_id="c9", ref_id="SRC001/12"),),
    )
    base.update(overrides)
    return RelevanceJudgment(**base)


# ── The value object ────────────────────────────────────────────────────


def test_a_judgment_ruling_on_nothing_states_nothing() -> None:
    assert _judgment(relevant=(), irrelevant=()).is_empty is True
    assert _judgment().is_empty is False


def test_round_trip_through_a_dict_preserves_every_field() -> None:
    original = _judgment(note="the reviewer flagged an omission", author="a@b.c")
    assert judgment_from_dict(judgment_to_dict(original)) == original


def test_a_record_written_by_an_older_version_still_loads() -> None:
    # The file is append-only and hand-editable, so an absent key must not
    # take the whole file down.
    loaded = judgment_from_dict({"id": "j2", "question": "a question"})
    assert loaded.id == "j2"
    assert loaded.relevant == ()
    assert loaded.status == "active"


# ── First use: the judgment becomes a permanent test ────────────────────


def test_relevant_chunks_become_a_golden_question() -> None:
    question = to_golden_question(_judgment())
    assert question is not None
    assert question["question"] == "What is the appeal deadline?"
    assert question["article_refs"] == ["SRC001/44"]
    assert question["judgment_id"] == "j1"


def test_one_source_unit_spanning_several_chunks_yields_one_ref() -> None:
    judgment = _judgment(relevant=(
        JudgedChunk(chunk_id="c1", ref_id="S/1"),
        JudgedChunk(chunk_id="c2", ref_id="S/1"),
        JudgedChunk(chunk_id="c3", ref_id="S/2"),
    ))
    assert to_golden_question(judgment)["article_refs"] == ["S/1", "S/2"]


def test_a_judgment_without_ref_ids_yields_no_test_rather_than_an_empty_one() -> None:
    """A question with empty article_refs classifies out_of_scope and can
    therefore never fail. That is worse than no test, so none is emitted."""
    judgment = _judgment(relevant=(JudgedChunk(chunk_id="c1"),))
    assert to_golden_question(judgment) is None


# ── Second use: training material ───────────────────────────────────────


def test_relevant_and_irrelevant_chunks_form_preference_pairs() -> None:
    assert preference_pairs(_judgment()) == [
        ("What is the appeal deadline?", "c1", "c9"),
    ]


def test_a_one_sided_judgment_yields_no_pair() -> None:
    assert preference_pairs(_judgment(irrelevant=())) == []


# ── The file ────────────────────────────────────────────────────────────


def test_one_file_per_realm_and_corpus(tmp_path: Path) -> None:
    a = judgments_path(tmp_path, "acme", "handbook_01")
    b = judgments_path(tmp_path, "demo", "handbook_01")
    assert a != b
    assert a.name == "acme__handbook_01.jsonl"


def test_ids_from_api_input_cannot_escape_the_directory(tmp_path: Path) -> None:
    """realm_id and corpus_id arrive from HTTP input. The separator is what
    makes an escape possible, so the property that matters is that the
    resolved path stays inside the directory, not that the name looks tidy —
    a dot survives sanitising and is harmless once no separator remains."""
    for realm, corpus in (("../../etc", "passwd"), ("..", ".."), ("a/b", "c\\d")):
        path = judgments_path(tmp_path, realm, corpus)
        assert path.parent == tmp_path
        assert path.resolve().parent == tmp_path.resolve()
        assert path.name.endswith(".jsonl")


def test_no_file_yet_is_an_ordinary_state_not_an_error(tmp_path: Path) -> None:
    assert load_judgments(tmp_path / "absent.jsonl") == []


def test_appending_preserves_what_was_already_recorded(tmp_path: Path) -> None:
    path = tmp_path / "j.jsonl"
    append_judgment(path, _judgment(id="j1"))
    append_judgment(path, _judgment(id="j2"))
    assert [j.id for j in load_judgments(path)] == ["j1", "j2"]


def test_one_bad_line_does_not_cost_every_other_judgment(tmp_path: Path) -> None:
    path = tmp_path / "j.jsonl"
    append_judgment(path, _judgment(id="j1"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{ this is not json\n")
    append_judgment(path, _judgment(id="j2"))
    assert [j.id for j in load_judgments(path)] == ["j1", "j2"]


def test_rewrite_replaces_the_file_and_leaves_no_temporary_behind(tmp_path: Path) -> None:
    path = tmp_path / "j.jsonl"
    append_judgment(path, _judgment(id="j1"))
    rewrite_judgments(path, [_judgment(id="j2")])
    assert [j.id for j in load_judgments(path)] == ["j2"]
    assert list(tmp_path.iterdir()) == [path]
