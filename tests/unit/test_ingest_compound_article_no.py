"""services/ingestion/cli.py:_read_file — article_no must recognize compound
numbers (КоАП "4.7", ТК "16.9", ...), not just plain integers. Before this
fix, .isdigit() silently dropped these to article_no=None, removing them
from retrieval_recall_at_k matching with no error (a legal corpus
integration: many codes use dotted compound article numbers).
"""
from __future__ import annotations

from pathlib import Path

from services.ingestion.cli import _read_file


def test_plain_integer_article_no(tmp_path: Path) -> None:
    f = tmp_path / "SRC001" / "5.txt"
    f.parent.mkdir(parents=True)
    f.write_text("Статья 5. Title\n\nBody.", encoding="utf-8")

    doc = _read_file(f)

    assert doc.metadata["article_no"] == "5"
    assert doc.metadata["source_code"] == "SRC001"


def test_dotted_compound_article_no(tmp_path: Path) -> None:
    f = tmp_path / "SRC006" / "4.7.txt"
    f.parent.mkdir(parents=True)
    f.write_text("Статья 4.7. Title\n\nBody.", encoding="utf-8")

    doc = _read_file(f)

    assert doc.metadata["article_no"] == "4.7"


def test_non_numeric_stem_yields_none(tmp_path: Path) -> None:
    f = tmp_path / "SRC001" / "kodeks_polnyy.txt"
    f.parent.mkdir(parents=True)
    f.write_text("some content", encoding="utf-8")

    doc = _read_file(f)

    assert doc.metadata["article_no"] is None
