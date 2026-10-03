"""The golden sets about the corpus as a whole agree with the corpus.

Their questions were drafted by a model and read by a person, and the
statements a right answer must make were then written out in one wording per
kind of fact. A wording typed by hand can carry a wrong number, and a wrong
statement in a golden set is a measurement that punishes the right answer. So
every statement is read back against the table the corpus is built from
(`tools/build_ground_corpus.py`), and against the document it names.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from tools.build_ground_corpus import MODELS

ROOT = Path(__file__).resolve().parents[2]
BY_CODE = {m.code: m for m in MODELS}
#: The first fifteen documents are the common procedure; a model's own card
#: follows, in the order of the table.
CARD = {m.code: f"{16 + i:02d}" for i, m in enumerate(MODELS)}

#: One wording per kind of fact and language: the pattern, and what the table
#: says the captured numbers must be.
FACTS: dict[str, list[tuple[str, Any]]] = {
    "base-ru": [
        (r"К обслуживанию (\S+) допускается инженер с допуском уровня (\d+)\.",
         lambda m: (m.admission_level,)),
        (r"Поверка (\S+) выполняется раз в (\d+) мес\.", lambda m: (m.verification_months,)),
        (r"Калибровка (\S+) выполняется раз в (\d+) мес\.", lambda m: (m.calibration_months,)),
        (r"(\S+) хранится при температуре от (-?\d+) до (-?\d+) градусов\.",
         lambda m: (m.storage_low, m.storage_high)),
        (r"Отчёт об обслуживании (\S+) сдаётся в течение (\d+) рабочих дней после выезда\.",
         lambda m: (m.report_days,)),
        (r"Основная расходная часть (\S+) заменяется раз в (\d+) мес\.",
         lambda m: (m.consumable_months,)),
    ],
    "base-en": [
        (r"Servicing the (\S+) requires an engineer authorised to level (\d+)\.",
         lambda m: (m.admission_level,)),
        (r"The (\S+) is stored between (-?\d+) and (-?\d+) degrees\.",
         lambda m: (m.storage_low, m.storage_high)),
        (r"The service report for the (\S+) is due within (\d+) working days of the visit\.",
         lambda m: (m.report_days,)),
        (r"The main consumable of the (\S+) is replaced every (\d+) months\.",
         lambda m: (m.consumable_months,)),
    ],
}


def rows_of(corpus: str) -> list[dict[str, Any]]:
    path = ROOT / "eval" / "golden" / f"{corpus}-global.v1.fast.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def disagreements(rows: list[dict[str, Any]], corpus: str) -> list[str]:
    """Everything a set says that the corpus does not."""
    found = []
    for row in rows:
        refs, statements = row.get("article_refs") or [], row.get("assertions") or []
        if len(refs) != len(statements):
            found.append(f"{row['id']}: {len(statements)} statements for {len(refs)} documents")
            continue
        for ref, statement in zip(refs, statements, strict=True):
            for pattern, expected in FACTS[corpus]:
                match = re.fullmatch(pattern, statement)
                if not match:
                    continue
                code, *numbers = match.groups()
                model = BY_CODE.get(code)
                if model is None:
                    found.append(f"{row['id']}: no model {code!r} in the corpus")
                elif tuple(int(n) for n in numbers) != expected(model):
                    found.append(f"{row['id']}: {statement!r} against {expected(model)} in the table")
                elif ref != f"{corpus}/{CARD[code]}":
                    found.append(f"{row['id']}: {code} is described in {corpus}/{CARD[code]}, "
                                 f"and the statement is attributed to {ref}")
                break
            else:
                found.append(f"{row['id']}: a statement in no known wording: {statement!r}")
    return found


@pytest.mark.parametrize("corpus", sorted(FACTS))
def test_every_statement_agrees_with_the_table_the_corpus_is_built_from(corpus: str) -> None:
    assert disagreements(rows_of(corpus), corpus) == []


@pytest.mark.parametrize("corpus", sorted(FACTS))
def test_every_question_is_about_the_corpus_as_a_whole_and_names_its_documents(corpus: str) -> None:
    rows = rows_of(corpus)
    assert len(rows) >= 5, "too few questions to tell one system from another"
    for row in rows:
        assert row["scope"] == "global" and row["ground_truth"].strip()
        assert len(set(row["article_refs"])) >= 3, f"{row['id']} draws on fewer than three documents"
        for ref in row["article_refs"]:
            number = ref.split("/")[1]
            assert (ROOT / "corpus" / "proving-ground" / corpus / f"{number}.md").is_file(), ref


@pytest.mark.parametrize("corpus", sorted(FACTS))
def test_a_question_that_names_no_instrument_lists_every_one_that_qualifies(corpus: str) -> None:
    """Some questions ask which instruments share a property, and their answer
    is wider than a context of five fragments, which is the point of them. A
    list missing one instrument would mark down the answer that names it, so
    the list is compared with the table whole."""
    wide = [row for row in rows_of(corpus) if row["id"].startswith("pg_corpus_wide_")]
    assert len(wide) >= 3, "no question here is wider than one context"
    for row in wide:
        assert len(row["assertions"]) > 5, f"{row['id']} fits in one context of five"
        named, shared = set(), set()
        for statement in row["assertions"]:
            for pattern, _ in FACTS[corpus]:
                match = re.fullmatch(pattern, statement)
                if match:
                    named.add(match.group(1))
                    shared.add((pattern, match.groups()[1:]))
                    break
        assert len(shared) == 1, f"{row['id']} asks about more than one property"
        (pattern, numbers), = shared
        expected = next(e for p, e in FACTS[corpus] if p == pattern)
        qualifying = {m.code for m in MODELS if expected(m) == tuple(int(n) for n in numbers)}
        assert named == qualifying, (
            f"{row['id']}: missing {sorted(qualifying - named)}, extra {sorted(named - qualifying)}")


def test_the_card_numbers_assumed_here_are_the_corpus_s_own() -> None:
    """The check above places a model's card by its position in the table. If
    the corpus were rebuilt in another order, every attribution would be
    judged against the wrong document, so the heading of each card is read."""
    for code, number in CARD.items():
        heading = (ROOT / "corpus" / "proving-ground" / "base-ru" / f"{number}.md").read_text(
            encoding="utf-8").splitlines()[0]
        assert code in heading, f"{number}.md is headed {heading!r}, not {code}"


# ── The check itself is checked ───────────────────────────────────────────────

def _one_row(statement: str, ref: str = "base-ru/22") -> list[dict[str, Any]]:
    return [{"id": "r", "article_refs": [ref], "assertions": [statement]}]


def test_a_right_statement_passes() -> None:
    assert disagreements(_one_row("Поверка ТМ-410 выполняется раз в 36 мес."), "base-ru") == []


def test_a_wrong_number_is_reported() -> None:
    assert disagreements(_one_row("Поверка ТМ-410 выполняется раз в 12 мес."), "base-ru")


def test_a_right_fact_attributed_to_another_document_is_reported() -> None:
    assert disagreements(_one_row("Поверка ТМ-410 выполняется раз в 36 мес.", "base-ru/23"), "base-ru")


def test_a_statement_in_a_wording_nobody_can_check_is_reported() -> None:
    """Passing what cannot be read would make the check as wide as its
    patterns and no wider, silently."""
    assert disagreements(_one_row("ТМ-410 is a fine instrument."), "base-ru")
