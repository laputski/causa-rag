import json
from pathlib import Path

from eval.dataset import EvalDataset, make_stub_dataset


def test_from_list():
    ds = EvalDataset.from_list([{"id": "q1", "question": "Q?", "reference_answer": "A"}])
    assert len(ds) == 1
    assert ds.questions[0]["question"] == "Q?"


def test_stub_dataset():
    ds = make_stub_dataset(n=10)
    assert len(ds) == 10
    assert ds.speed == "fast"


def test_subset():
    ds = make_stub_dataset(n=20)
    sub = ds.subset(5)
    assert len(sub) == 5
    assert sub.name == ds.name


def test_save_and_load_jsonl(tmp_path: Path):
    ds = make_stub_dataset(n=3)
    path = tmp_path / "test.v1.fast.jsonl"
    ds.save_jsonl(path)
    loaded = EvalDataset.from_jsonl(path)
    assert len(loaded) == 3
    assert loaded.name == "test"
    assert loaded.version == "v1"
    assert loaded.speed == "fast"


def test_from_jsonl_parses_questions(tmp_path: Path):
    path = tmp_path / "stub.v0.fast.jsonl"
    questions = [{"id": f"q{i}", "question": f"Q{i}?", "reference_answer": f"A{i}"} for i in range(5)]
    path.write_text("\n".join(json.dumps(q) for q in questions), encoding="utf-8")
    ds = EvalDataset.from_jsonl(path)
    assert len(ds) == 5
    assert ds.questions[2]["id"] == "q2"


def test_empty_lines_ignored(tmp_path: Path):
    path = tmp_path / "ds.v1.fast.jsonl"
    path.write_text('{"id":"q1","question":"Q?","reference_answer":"A"}\n\n\n', encoding="utf-8")
    ds = EvalDataset.from_jsonl(path)
    assert len(ds) == 1
