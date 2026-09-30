"""Checking an answer against the statements it must make.

A question may carry assertions: short statements a correct answer has to
contain. A judge, which is a language model other than the one that answered,
decides for each whether the answer states it. This module builds the judge's
prompt and reads its reply; calling the model is the caller's business, the way
it is for `core/eval/triage.py`.

What was measured before the prompt was written, on the proving ground's two
corpora with hand-written pairs (a statement, an answer that makes it, an answer
that does not):

- One statement per call. Four statements in one call were judged worse, in
  English too, where one at a time was judged without error.
- The rule about equivalent quantities is needed in Russian: without it the
  judge rejected correct answers that said "две недели" for "четырнадцать
  дней". With it, and with examples that appear in no test case, a small
  model began accepting wrong conversions as well, and a larger one did not.
- A reply that cannot be read is no verdict. It is never a "not stated",
  because a zero the judge did not give is a number nobody measured.
"""
from __future__ import annotations

import json
import re
from typing import Any

_PROMPT_RU = """Ты проверяешь, высказано ли в ответе утверждение.

Утверждение высказано, если ответ говорит то же самое, в любой формулировке и
на любом языке. Одна и та же величина, выраженная другими единицами или
словами, считается тем же самым: например, «полчаса» и «тридцать минут»,
«двое суток» и «сорок восемь часов», «декада» и «десять дней».

Утверждение не высказано, если ответ его опускает, высказывает лишь часть его
или говорит другое: другое число или срок после перевода единиц, другое лицо,
другое условие.

Ответ:
<<<
{answer}
>>>

Утверждение:
{assertion}

Верни только JSON: {{"stated": true или false}}"""

_PROMPT_EN = """You check whether an answer states an assertion.

The assertion is stated if the answer says the same thing, in any wording or
language. One quantity expressed in other units or words counts as the same:
for example "half an hour" and "thirty minutes", "two days" and "forty-eight
hours", "a fortnight" and "fourteen days".

The assertion is not stated if the answer omits it, states only part of it,
or says something different: a different number or period after converting
units, a different person, a different condition.

Answer:
<<<
{answer}
>>>

Assertion:
{assertion}

Return JSON only: {{"stated": true or false}}"""

#: Why an answer carrying assertions went without a verdict. A closed set, so
#: a finding can carry the reason as an identifier and the reader gets a word
#: in their own language; a library's message, when there is one, travels
#: beside it as it arrived.
JUDGE_GAPS = ("unreachable", "model_not_pulled", "transport_failed", "prompts_cut", "unparsed")

_CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)
_LETTER = re.compile(r"[^\W\d_]")


def language_of(statement: str) -> str:
    """The prompt follows the statement's language and not the answer's.

    An answer in another language stays checkable, and the rule about
    equivalent quantities is written for the language the statement is in.
    """
    letters = _LETTER.findall(statement)
    if not letters:
        return "en"
    return "ru" if len(_CYRILLIC.findall(statement)) / len(letters) > 0.3 else "en"


def build_prompt(answer: str, assertion: str) -> str:
    template = _PROMPT_RU if language_of(assertion) == "ru" else _PROMPT_EN
    return template.format(answer=answer, assertion=assertion)


def parse_verdict(reply: str) -> bool | None:
    """True or False when the judge said so, None when it said nothing
    readable."""
    try:
        value = json.loads(reply).get("stated")
    except (ValueError, AttributeError):
        return None
    return value if isinstance(value, bool) else None


def coverage(verdicts: list[dict[str, Any]]) -> float | None:
    """The share of statements the answer makes, or None when any verdict is
    missing. A share over the statements that happened to be judged would be
    a number about another question than the one asked."""
    if not verdicts or any(v.get("stated") is None for v in verdicts):
        return None
    return sum(1 for v in verdicts if v["stated"]) / len(verdicts)
