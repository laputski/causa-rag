"""Extend the proving ground's corpora with a family of instrument service cards.

The first fifteen documents of each corpus are general procedure: what to do on
receiving an instrument, how often to calibrate, who may sign the log. They are
enough to answer a question and not enough to make a question hard. Measured on
them, the expected source ranked first on every question, the two retrieval
halves agreed on sixty of seventy-five returned sources, and a candidate window
of fifty covered the whole corpus. Three catalogue entries could not be staged
there at all: one needs the halves to disagree, one needs the window to be
narrower than the corpus, and one needs a question phrased in words the
document does not use.

What follows is what a real handbook has beside its general procedure: one card
per instrument model, each repeating the same six headings with that model's
own numbers. That shape is what creates competition. A question about
calibration now matches thirty documents on almost every word, and the one that
answers it differs from the rest by a model code and a figure.

Written as a table and six templates, and not as three hundred paragraphs by
hand: the table is the part that carries meaning and is reviewable as a table,
and a template repeated with different values is exactly what the source
documents of this kind look like. Every section carries at least one value of
its own, so no two sections are ever identical, which the health check would
otherwise read as a duplicated corpus. That property is asserted by
`tests/unit/test_ground_corpus_builder.py` and not by this sentence: the first
version of the card opened with a preamble that named no model, and the health
check reported twenty-five duplicates on a corpus this file described as
carrying none.

    python3 -m tools.build_ground_corpus            # write both corpora
    python3 -m tools.build_ground_corpus --limit 8  # a sample, to measure first
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GROUND = REPO_ROOT / "corpus" / "proving-ground"
FIRST_NUMBER = 16


@dataclass(frozen=True)
class Model:
    """One instrument, and everything the six sections differ by."""

    code: str
    family_ru: str
    family_en: str
    verification_months: int
    calibration_months: int
    consumable_ru: str
    consumable_en: str
    consumable_months: int
    admission_level: int
    storage_low: int
    storage_high: int
    report_days: int


MODELS: tuple[Model, ...] = (
    Model("АТ-120", "титратор", "titrator", 12, 3, "мембрана электрода", "electrode membrane", 6, 2, 5, 35, 3),
    Model("АТ-140", "титратор", "titrator", 12, 4, "мембрана электрода", "electrode membrane", 9, 2, 5, 35, 3),
    Model("СФ-210", "спектрофотометр", "spectrophotometer", 24, 6, "дейтериевая лампа", "deuterium lamp", 12, 3, 10, 30, 5),
    Model("СФ-260", "спектрофотометр", "spectrophotometer", 24, 3, "дейтериевая лампа", "deuterium lamp", 18, 3, 10, 30, 5),
    Model("ХР-310", "хроматограф", "chromatograph", 12, 1, "разделительная колонка", "separation column", 24, 4, 15, 28, 10),
    Model("ХР-350", "хроматограф", "chromatograph", 12, 2, "разделительная колонка", "separation column", 18, 4, 15, 28, 10),
    Model("ТМ-410", "термостат", "thermostat", 36, 12, "воздушный фильтр", "air filter", 6, 1, 0, 40, 2),
    Model("ТМ-450", "термостат", "thermostat", 36, 6, "воздушный фильтр", "air filter", 3, 1, 0, 40, 2),
    Model("ВС-510", "весы", "balance", 12, 1, "калибровочная гиря", "calibration weight", 36, 2, 15, 25, 3),
    Model("ВС-560", "весы", "balance", 12, 2, "калибровочная гиря", "calibration weight", 24, 2, 15, 25, 3),
    Model("НС-610", "насос", "pump", 24, 12, "уплотнение вала", "shaft seal", 12, 1, -5, 45, 2),
    Model("НС-650", "насос", "pump", 24, 6, "уплотнение вала", "shaft seal", 9, 1, -5, 45, 2),
    Model("ДТ-710", "детектор", "detector", 12, 3, "оптическое окно", "optical window", 12, 3, 10, 32, 5),
    Model("ДТ-760", "детектор", "detector", 12, 6, "оптическое окно", "optical window", 18, 3, 10, 32, 5),
    Model("ПР-810", "пробоотборник", "sampler", 36, 12, "игла дозатора", "dosing needle", 6, 2, 5, 38, 3),
    Model("ПР-860", "пробоотборник", "sampler", 36, 6, "игла дозатора", "dosing needle", 9, 2, 5, 38, 3),
    Model("КЛ-910", "калибратор", "calibrator", 12, 2, "эталонная ячейка", "reference cell", 36, 4, 18, 24, 10),
    Model("КЛ-960", "калибратор", "calibrator", 12, 1, "эталонная ячейка", "reference cell", 24, 4, 18, 24, 10),
    Model("МШ-020", "мешалка", "stirrer", 36, 24, "приводной ремень", "drive belt", 12, 1, 0, 45, 2),
    Model("МШ-070", "мешалка", "stirrer", 36, 12, "приводной ремень", "drive belt", 18, 1, 0, 45, 2),
    Model("ЦФ-130", "центрифуга", "centrifuge", 12, 6, "уплотнение крышки", "lid gasket", 12, 3, 5, 35, 5),
    Model("ЦФ-180", "центрифуга", "centrifuge", 12, 3, "уплотнение крышки", "lid gasket", 9, 3, 5, 35, 5),
    Model("ИК-240", "анализатор", "analyser", 24, 4, "кювета сравнения", "reference cuvette", 18, 4, 12, 30, 5),
    Model("ИК-290", "анализатор", "analyser", 24, 2, "кювета сравнения", "reference cuvette", 12, 4, 12, 30, 5),
    Model("ЭЛ-330", "электрометр", "electrometer", 12, 12, "изолирующая втулка", "insulating bush", 24, 2, 8, 33, 3),
)


def _document_ru(number: int, m: Model) -> str:
    return f"""# {number} Карта обслуживания {m.code}

Карта относится к модели {m.code} и ни к какой другой. Общий порядок для всех приборов изложен в разделах с первого по пятнадцатый и этой картой не отменяется.

## Назначение

{m.code} есть {m.family_ru} для лабораторных измерений. Карта действует для всех исполнений модели {m.code} независимо от года выпуска.

## Поверка и калибровка

Поверка {m.code} выполняется раз в {m.verification_months} мес. Калибровка {m.code} выполняется раз в {m.calibration_months} мес и после всякой замены измерительного модуля.

## Расходные части

Основная расходная часть {m.code} называется так: {m.consumable_ru}. Замена выполняется раз в {m.consumable_months} мес либо по показаниям самодиагностики.

## Допуск персонала

К обслуживанию {m.code} допускается инженер с допуском уровня {m.admission_level}. Допуск ниже уровня {m.admission_level} даёт право только на внешний осмотр.

## Хранение

{m.code} хранится при температуре от {m.storage_low} до {m.storage_high} градусов. Выход за этот предел требует внеочередной калибровки перед вводом в работу.

## Отчётность

Отчёт об обслуживании {m.code} сдаётся в течение {m.report_days} рабочих дней после выезда. Отчёт содержит заводской номер и перечень заменённых частей.
"""


def _document_en(number: int, m: Model) -> str:
    return f"""# {number} Service card for the {m.code}

This card covers the {m.code} and no other model. The general procedure for every instrument is set out in sections one to fifteen and is not replaced by this card.

## What it is

The {m.code} is a laboratory {m.family_en}. The card applies to every build of the {m.code} whatever its year of manufacture.

## Verification and calibration

The {m.code} is verified every {m.verification_months} months. The {m.code} is calibrated every {m.calibration_months} months and after any replacement of the measuring module.

## Consumable parts

The main consumable of the {m.code} is the {m.consumable_en}. It is replaced every {m.consumable_months} months, or sooner if self-diagnosis calls for it.

## Personnel authorisation

Servicing the {m.code} requires an engineer authorised to level {m.admission_level}. An authorisation below level {m.admission_level} carries the right to external inspection only.

## Storage

The {m.code} is stored between {m.storage_low} and {m.storage_high} degrees. Going outside that range calls for an unscheduled calibration before the instrument is put back to work.

## Reporting

The service report for the {m.code} is due within {m.report_days} working days of the visit. The report carries the serial number and the list of parts replaced.
"""


def build(limit: int | None = None, destination: Path | None = None) -> dict[str, int]:
    written: dict[str, int] = {}
    models = MODELS[:limit] if limit else MODELS
    root = destination or GROUND
    for corpus_id, render in (("base-ru", _document_ru), ("base-en", _document_en)):
        directory = root / corpus_id
        directory.mkdir(parents=True, exist_ok=True)
        for offset, model in enumerate(models):
            number = FIRST_NUMBER + offset
            (directory / f"{number}.md").write_text(render(number, model), encoding="utf-8")
        written[corpus_id] = len(models)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--limit", type=int, default=None, help="write only the first N models")
    parser.add_argument("--out", type=Path, default=None, help="write somewhere other than the corpus")
    args = parser.parse_args(argv)
    for corpus_id, count in build(args.limit, args.out).items():
        print(f"{corpus_id}: {count} service cards, numbered from {FIRST_NUMBER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
