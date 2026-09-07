"""Put one known defect into a healthy corpus, and nothing else.

A proving ground is only as good as the difference between its two halves. Hand
written variants drift: somebody edits the healthy corpus, forgets the broken
copy, and the bait that was proving "the guard reddens on this defect" quietly
starts proving that two unrelated corpora differ. So the broken half is derived
from the healthy one, by a named change, on demand.

That makes this tool serve three purposes at once. It builds the corpora the
proving ground runs on; it generates the fixtures the unit-level baits use, so
those cannot drift from what ingestion really produces either; and it is the
thing that turns "differ by exactly one defect" from an intention into a fact.

The guarantee has a boundary worth stating. One defect in the **source files**
is what this promises. After ingestion the two indexes differ in everything
downstream of the changed bytes, chunk identifiers and vectors included, and
that is the point: those differences are what the guards read.

    python3 -m tools.corpus_mutate corpus/demo_handbook --defect flatten_headings --out /tmp/broken
    python3 -m tools.corpus_mutate --list
"""
from __future__ import annotations

import argparse
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# A corpus as this tool sees it: file name to text. Read whole, because a defect
# may need to look across documents (a number repeated in two of them) and
# because a corpus small enough to be a proving ground fits in memory by design.
Corpus = dict[str, str]


@dataclass(frozen=True)
class Defect:
    """One named way to break a corpus, and what it is meant to provoke."""

    name: str
    #: What the corpus looks like afterwards, in the words a reader would use.
    describes: str
    #: The catalogue entries this defect is meant to make observable. Named so a
    #: mutation nobody has a use for is visible as such.
    provokes: tuple[str, ...]
    apply: Callable[[Corpus], Corpus]
    #: What the input corpus must already be for this defect to exist in it, and
    #: an empty string when anything will do.
    #:
    #: Three defects were written without this and produced a corpus that looked
    #: mutated and provoked nothing: two of them repeat or drop a *structural*
    #: number, and the corpus they were run against numbers its files and not its
    #: headings, which is what the health check reads. A mutation that silently
    #: changes nothing observable is worse than one that refuses.
    requires: str = ""
    #: Whether the corpus can carry this defect at all.
    admits: Callable[[Corpus], bool] | None = None


def _has_numbered_headings(corpus: Corpus) -> bool:
    numbered = re.compile(r"^#{1,6}\s+\d+(\.\d+)*[.\s]", re.M)
    return any(numbered.search(text) for text in corpus.values())


def _flatten_headings(corpus: Corpus) -> Corpus:
    """Strip the markdown headings, keeping every word of the body.

    The structural strategy then has nothing to read and falls back to flat
    windows, which is what "the chosen segmentation strategy silently does
    nothing" looks like from the outside. The demo realm shipped in this state
    for the whole time the incident log said it did not.
    """
    return {
        name: "\n".join(
            line for line in text.splitlines() if not line.lstrip().startswith("#")
        )
        for name, text in corpus.items()
    }


def _duplicate_documents(corpus: Corpus) -> Corpus:
    """Copy every document under a second name, byte for byte.

    What a second ingest of the same files produces when a chunk identifier is
    not derived from content: the index grows instead of being updated.
    """
    out = dict(corpus)
    for name, text in corpus.items():
        stem, _, suffix = name.rpartition(".")
        out[f"{stem}-copy.{suffix}"] = text
    return out


def _drop_a_numbered_document(corpus: Corpus) -> Corpus:
    """Remove one document from the middle of a sequential numbering.

    A gap that the source has and the index does not: the unit exists on disk
    and reached nothing, which is what a lost export looks like afterwards.
    """
    numbered = sorted(n for n in corpus if re.fullmatch(r"\d+\.\w+", n))
    if len(numbered) < 3:
        return dict(corpus)
    victim = numbered[len(numbered) // 2]
    return {name: text for name, text in corpus.items() if name != victim}


def _repeat_a_structural_number(corpus: Corpus) -> Corpus:
    """Add a document carrying a number the corpus already uses.

    The signature of an export that gave one number to two different units: the
    number is there twice under two different headings, and a reader cannot tell
    which of them the index answered with. It is the shape of a real incident,
    where two consecutive units of a source both came out numbered the same and
    one of them reached nothing.

    Added and never renumbered. Two earlier versions moved a number from one
    document to another, and moving a number leaves a hole where it was, so the
    corpus carried a duplicate *and* a gap and the mutation provoked two checks
    at once. Adding leaves every existing number where it is.
    """
    numbered = sorted(n for n in corpus if re.fullmatch(r"\d+\.\w+", n))
    if not numbered:
        return dict(corpus)
    lead = re.compile(r"^(#{1,6}\s+)(\d+(?:\.\d+)*)(\s)(.*)$", re.M)
    donor = lead.search(corpus[numbered[0]])
    if donor is None:
        return dict(corpus)
    out = dict(corpus)
    stem, _, suffix = numbered[0].rpartition(".")
    out[f"{stem}-second.{suffix}"] = (
        f"{donor.group(1)}{donor.group(2)} {donor.group(4)} (вторая редакция)\n\n"
        "Этот раздел вышел из выгрузки под тем же номером, что и предыдущий, и "
        "содержит совершенно другое правило, относящееся к другому случаю.\n"
    )
    return out


#: Below this a chunk carries too little to answer from, and it is the health
#: check's own threshold and never a number chosen here: a mutation aiming
#: above it produces a corpus that reads as damaged and is reported as healthy.
_TOO_SHORT_BELOW = 50


def _shrink_to_fragments(corpus: Corpus) -> Corpus:
    """Leave each section a single short line.

    Chunks too small to carry the grounds for an answer: retrieval finds the
    right unit and the unit says too little to answer from.

    Shrinking *paragraphs* was the first two attempts and neither worked, for
    two different reasons worth keeping. Cutting to the first sentence left an
    average of two hundred characters against a threshold of fifty, so the
    corpus read as damaged and every check called it healthy. Cutting to four
    words dropped chunks under forty characters, where a different check claims
    them as carrying a heading and no body, so the defect provoked the wrong
    signal.

    Neither worked at all in the end, because the chunker packs short paragraphs
    back together up to its own size: shortening the parts of a section does not
    shorten the section. The unit that becomes a chunk is the section, so that
    is the unit to shrink.
    """
    heading = re.compile(r"^#{1,6}\s")

    def shrink(text: str) -> str:
        out: list[str] = []
        body: list[str] = []

        def flush() -> None:
            if not body:
                return
            words, taken = " ".join(body).split(), []
            while words and len(" ".join([*taken, words[0]])) < _TOO_SHORT_BELOW - 4:
                taken.append(words.pop(0))
            if taken:
                out.extend(["", " ".join(taken) + "."])
            body.clear()

        for line in text.splitlines():
            if heading.match(line):
                flush()
                out.append(line)
            elif line.strip():
                body.append(line.strip())
        flush()
        return "\n".join(out) + "\n"

    return {name: shrink(text) for name, text in corpus.items()}


# Two short documents in each language the proving ground uses, wholly distinct
# from one another. An earlier version shared an opening between them, and the
# shared part chunked identically, so the duplicate check fired beside the
# language one: a bait proving two things at once proves neither.
#: Documents of the other language, as templates over their own number.
#:
#: Every body carries that number, so two copies are never the same text: the
#: first version repeated a fixed pair and, once enough copies were needed to
#: cross the language threshold, the copies themselves read as duplicates and
#: the mutation provoked two signals instead of the one it names.
#:
#: The sub-headings carry no number, and that is deliberate. The structural
#: check reads the first number out of a heading, so a sub-heading reading
#: "Thresholds of revision 1" was a second unit numbered one, colliding with
#: the corpus's own first document and provoking a third signal. The top-level
#: number is prefixed with a nine and cannot collide with a corpus numbered
#: from one.
_OTHER_LANGUAGE: dict[str, tuple[str, ...]] = {
    "ru": (
        "# 9{n} Порядок согласования, редакция {n}\n\n"
        "Редакция {n} определяет, кто согласует закупку и какие подтверждения требуются "
        "на каждом уровне.\n\n"
        "## Пороговые значения\n\n"
        "В редакции {n} уровень В начинается с пятидесяти тысяч и утверждается "
        "финансовым директором.\n",
        "# 9{n} Командировки и расходы, редакция {n}\n\n"
        "По редакции {n} суточные выплачиваются за каждый полный день пребывания вне "
        "места работы.\n\n"
        "## День отъезда\n\n"
        "В редакции {n} день отъезда и день возвращения оплачиваются в половинном "
        "размере.\n",
    ),
    "en": (
        "# 9{n} Approval procedure, revision {n}\n\n"
        "Revision {n} defines who approves a purchase and what evidence each level "
        "requires.\n\n"
        "## Thresholds\n\n"
        "Under revision {n} band C starts at fifty thousand and is approved by the "
        "finance director.\n",
        "# 9{n} Travel and expenses, revision {n}\n\n"
        "Under revision {n} a daily allowance is paid for every full day spent away "
        "from the place of work.\n\n"
        "## The day of departure\n\n"
        "In revision {n} the day of departure and the day of return are paid at half "
        "that rate.\n",
    ),
}


def _dominant_language(corpus: Corpus) -> str:
    from langdetect import LangDetectException, detect

    for text in corpus.values():
        try:
            return detect(text)
        except LangDetectException:
            continue
    return "en"


def _add_a_second_language(corpus: Corpus) -> Corpus:
    """Put documents of another language beside the existing ones.

    Not a translation of anything present: the point is a corpus holding two
    languages at once, so a model or a reranker covering one of them is
    measurably worse on the other.

    Which language is added depends on which the corpus already is. The first
    version always added Russian, which on a Russian corpus is not a second
    language at all: the mutation ran, the corpus changed, and the language
    check correctly reported one language. A defect that cannot exist in the
    corpus it was applied to has to be visible as such.
    """
    own = _dominant_language(corpus)
    other = "en" if own.startswith("ru") else "ru"
    out = dict(corpus)
    # How many to add is derived from the corpus and no longer a constant two.
    # The health check flags a second language only above a share of five per
    # cent of the sampled chunks, so a fixed pair crosses that on a corpus of
    # fifteen documents and falls under it on one of forty: measured, the
    # mutation ran, the corpus changed, and every check called it healthy.
    # Two-fifths of the document count leaves a wide margin at either size.
    added = max(2, round(len(corpus) * 0.4))
    sources = _OTHER_LANGUAGE[other]
    for index in range(1, added + 1):
        template = sources[(index - 1) % len(sources)]
        out[f"9{index:02d}.md"] = template.format(n=index)
    return out


def _repeat_a_phrase_in_every_document(corpus: Corpus) -> Corpus:
    """Put one identical sentence under every heading of every document.

    What a handbook acquires when a template, a disclaimer or a page footer is
    pasted throughout. Nothing about it looks wrong: the sentence is the
    corpus's own prose, every document still reads correctly, and no length,
    number or heading moves.

    The sentence is lifted from the corpus and never written here, which is
    not a convenience. A phrase of this module's own choosing is a phrase in
    this module's own language, and a Russian sentence pasted into an English
    handbook stages a language failure and not a graph one. A guard found
    exactly that.

    Under *every* heading, and that is a measurement. Under the first only,
    the phrase reached forty-one of two hundred and twenty units, and the
    graph came back with fewer edges than the control: keywords are the first
    eight distinct words of a unit, so a sentence at the top of a unit
    displaces the unit's own keywords and does not add to them. Under every
    heading it reaches every unit, all eight keywords of every unit exceed the
    linker's frequency cap at once, and the link step produces nothing.

    So what this stages is not the runaway link step it was written for. It is
    that step's own prevention misfiring, which is a failure of its own and
    now an entry of its own.
    """
    phrase = _longest_sentence(corpus)
    if not phrase:
        return dict(corpus)
    return {name: _insert_under_every_heading(text, phrase) for name, text in corpus.items()}


def _longest_sentence(corpus: Corpus) -> str:
    """The longest ordinary sentence in the corpus, which carries the most
    keywords to share and is by construction in the corpus's own language."""
    best = ""
    for text in corpus.values():
        for line in text.splitlines():
            if line.startswith("#") or line.startswith("-") or line.startswith("|"):
                continue
            for sentence in re.split(r"(?<=[.!?])\s+", line):
                sentence = sentence.strip()
                if len(sentence) > len(best):
                    best = sentence
    return best


def _insert_under_every_heading(text: str, phrase: str) -> str:
    """Put the phrase under each heading, where a template would sit."""
    out: list[str] = []
    seen_heading = False
    for line in text.splitlines():
        out.append(line)
        if line.startswith("#"):
            seen_heading = True
            out += ["", phrase]
    if not seen_heading:
        return phrase + "\n\n" + text
    return "\n".join(out)


def _stuff_a_document_with_the_corpus_own_words(corpus: Corpus) -> Corpus:
    """Add one document that repeats the corpus's commonest words and says
    nothing.

    What a keyword index rewards and a reader would throw away: a page of
    boilerplate, an index, a glossary of headings, a table of contents pasted
    into a document of its own. It carries every word a question is likely to
    use and none of the meaning, so lexical search ranks it near the top of
    almost every query while semantic search leaves it where it belongs.

    The words come from the corpus and are never written here, for the same
    reason the repeated phrase is lifted and never composed: a word chosen
    by this module is a word in this module's language, and the questions are
    asked in the corpus's.
    """
    words = _commonest_words(corpus, count=12)
    if not words:
        return dict(corpus)
    filler = " ".join(words)
    out = dict(corpus)
    # Numbered past the end so it never displaces a document a question
    # refers to, and so the corpus's own numbering stays a run without gaps.
    name = f"{len(corpus) + 1:02d}.md"
    out[name] = (
        f"# {len(corpus) + 1} Указатель терминов\n\n"
        + "\n\n".join(f"{filler}." for _ in range(12))
        + "\n"
    )
    return out


def _commonest_words(corpus: Corpus, count: int) -> list[str]:
    """The corpus's own frequent content words, longest first among ties."""
    from collections import Counter

    seen: Counter[str] = Counter()
    for text in corpus.values():
        for word in re.findall(r"[^\W\d_]{6,}", text.lower()):
            seen[word] += 1
    return [word for word, _ in seen.most_common(count)]


def _leave_every_other_section_a_heading(corpus: Corpus) -> Corpus:
    """Replace the text under every other heading with the heading's own words.

    What a handbook becomes when an export walks the table of contents and
    loses the body of the sections it does not understand: the title, the
    numbering and the section list all survive, so a reader skimming the files
    sees a corpus in order.

    This is the failure that looks *most* like health from the retrieval side.
    A title is a dense statement of its own subject, so a question about that
    subject matches it better than most prose does: the search finds the right
    unit, at a good score, and the unit has nothing under it to answer from.

    Two measurements shaped it. Emptying a section outright leaves nothing at
    all, because the chunker drops a heading with no text under it, and the
    corpus came back empty: the mutation staged the wrong failure entirely.
    And doing it to *every* section drops the average length under fifty,
    where a different check calls the corpus short, so the defect provoked two
    signals and a pair on it would have proved neither. Every other section
    keeps more than a third of the units under the length that reads as a
    heading while the average stays where a healthy corpus has it.
    """
    heading = re.compile(r"^(#{1,6})\s+(.*)$")

    def strip(name: str, text: str) -> str:
        out: list[str] = []
        section = 0
        skipping = False
        stem = name.rsplit(".", 1)[0]
        for line in text.splitlines():
            match = heading.match(line)
            if match:
                section += 1
                skipping = section % 2 == 0
                out.append(line)
                if skipping:
                    # The heading's own words, and nothing else: what an export
                    # writes when it has the title of a section and not its
                    # body.
                    #
                    # Two measurements shaped the exact string. It is prefixed
                    # with where it came from because two documents of the
                    # proving ground's corpus carry a section under the same
                    # name, so the residues came out byte identical and the
                    # mutation staged a duplicate as well. And it is kept
                    # under the length at which the language check will look
                    # at a fragment at all: at twenty-six characters the
                    # English half's residues were read as another language
                    # and the mutation staged a second failure again. The
                    # check calls that length unreliable itself, which is why
                    # staying under it is the honest fix and not a dodge.
                    out.append("")
                    residue = f"{stem}.{section} {match.group(2)[:13].rstrip()}"
                    out.append(residue[:19])
                continue
            if not skipping:
                out.append(line)
        return "\n".join(out).strip() + "\n"

    return {name: strip(name, text) for name, text in corpus.items()}


def _empty_the_corpus(corpus: Corpus) -> Corpus:
    """Every document gone, and the directory still there.

    The extreme of a load that failed, a path that pointed at the wrong place,
    or a filter that matched nothing. It earns a defect of its own because of
    what the *numbers* do with it, which is the part nobody expects: every
    retrieval metric comes back at zero, and zero is also the shape of a
    catastrophically bad retriever. Two configurations compared against an
    empty index are equal, and the comparison reads as a finished measurement.
    """
    return {}


def _cut_a_table_and_a_list(corpus: Corpus) -> Corpus:
    """Add one document that is mostly a table and a long list.

    Splitting by size knows nothing of a table: it cuts between two rows, and
    the half carrying no header row is a grid of numbers whose columns have
    lost their names. A list is cut the same way, and the tail arrives without
    the sentence that said what the items are.

    The staged failure is not that the split happens; it is that nothing says
    so. The corpus stays well formed, every length, number and heading is in
    order, and the checks report health, so this defect's pair is the reverse
    kind: the defect is present and the honest expectation is silence.

    Every word here is the corpus's own, for the reason a sibling defect
    records: a sentence of this module's choosing is a sentence in this
    module's language, and the first version of this one was written in
    Russian and dropped into an English handbook, where it staged a language
    failure instead of a table one. Measured, not foreseen.
    """
    words = _commonest_words(corpus, count=10)
    if not words:
        return dict(corpus)
    out = dict(corpus)
    number = len(corpus) + 1
    columns = words[:4]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = "\n".join(
        "| " + " | ".join(f"{word} {i}" for word in columns) + " |"
        for i in range(1, 61)
    )
    prose = " ".join(words)
    items = "\n".join(f"- {prose}, {i}." for i in range(1, 61))
    out[f"{number:02d}.md"] = (
        f"# {number} {' '.join(words[:3])}\n\n"
        f"{prose}, and the table below sets each of them out in turn.\n\n"
        f"## {' '.join(words[3:6])}\n\n"
        f"{header}\n{divider}\n{rows}\n\n"
        f"## {' '.join(words[6:9])}\n\n"
        f"{items}\n"
    )
    return out


#: How many units share the injected word.
#:
#: The linker drops a keyword once it occurs in more units than its own cap,
#: which is fifty, so a word placed in every unit produces no edges at all and
#: stages that prevention instead of the runaway it was meant to. Fifty is the
#: largest number of units a shared word can join before the cap removes it,
#: and it is therefore where the runaway is at its worst: one bucket of fifty
#: contributes 50x49/2 edges by itself.
UNITS_SHARING_THE_WORD = 50


def share_a_word(corpus: Corpus, units: int) -> Corpus:
    """One invented word at the head of the body of `units` sections, spread
    one section per document at a time.

    Exposed beside the defect because the failure being staged is a *shape*
    and one measurement cannot show a shape. A pair at two sizes can: twice
    the units carrying the word, four times the edges it contributes. The
    defect below fixes the size; a caller measuring the growth varies it.

    Spread and never filled document by document, which is what makes the
    grouping it produces readable. Fifty units taken from the front are nine
    documents, and units of one document belong together anyway, so a
    community built from them says nothing. One unit per document, round by
    round, joins sections of forty unrelated documents into a clique that has
    the word and nothing else in common.

    At the head of the body, which is a measurement and not a preference.
    Keywords are the first eight distinct words of four characters or more,
    in the order they appear, so a word placed anywhere else in a long
    section is not a keyword of it and joins nothing.

    The word is the corpus's commonest doubled, so it is in the corpus's own
    script and is not a word of any language, which is what keeps it from
    colliding with a real keyword. A token written here would be a token in
    this module's language, and a sibling defect records what that cost.
    """
    words = _commonest_words(corpus, count=1)
    if not words:
        return dict(corpus)
    shared = words[0] * 2
    heading = re.compile(r"^#{1,6}\s")

    # Where each document's section bodies begin, so the rounds below can take
    # one section from each document in turn without re-parsing.
    bodies: dict[str, list[int]] = {}
    lines: dict[str, list[str]] = {}
    for name, text in sorted(corpus.items()):
        rows = text.splitlines()
        lines[name] = rows
        at: list[int] = []
        awaiting = False
        for i, row in enumerate(rows):
            if heading.match(row):
                awaiting = True
            elif awaiting and row.strip():
                at.append(i)
                awaiting = False
        bodies[name] = at

    left = units
    round_number = 0
    while left > 0 and any(len(at) > round_number for at in bodies.values()):
        for name, at in bodies.items():
            if left <= 0:
                break
            if len(at) <= round_number:
                continue
            i = at[round_number]
            lines[name][i] = f"{shared} {lines[name][i]}"
            left -= 1
        round_number += 1
    return {name: "\n".join(rows) + "\n" for name, rows in lines.items()}


def _share_a_word_at_the_cap(corpus: Corpus) -> Corpus:
    """A shared word in as many units as the linker's cap will still link.

    One word in fifty units puts all fifty in one keyword bucket, and a bucket
    of N contributes N(N-1)/2 edges: 1225 of them, on top of a corpus whose
    own linking produces about 3756. The corpus reads correctly, every length,
    number and heading is in order, and nothing about the documents is
    malformed. What grows is the graph.
    """
    return share_a_word(corpus, UNITS_SHARING_THE_WORD)


#: How long one section has to be before an embedding model stops reading it.
#:
#: Measured on the model this proving ground indexes with, by embedding a text
#: with and without a sentence at its end: at fifty thousand characters the
#: sentence stopped changing the vector at all, to six decimal places. The
#: window is declared as 8192 tokens and this is what that is in Russian prose.
#: Sixty thousand is past it with room, and still under a chunk size a load can
#: be given, which is what the other half of this pair needs.
CHARACTERS_PAST_THE_MODEL_WINDOW = 60_000


def _pad_a_section_past_the_model_window(corpus: Corpus) -> Corpus:
    """Add one document whose single section is longer than a model will read.

    A section, and not a document: a document of any length is cut into units
    and every unit is read. What is never read is the end of a *unit* longer
    than the window, and a unit is as long as the load's chunk size allows.
    So this half makes the text and the load's half makes the unit.

    The last sentence carries a fact that appears nowhere else in the corpus,
    which is how the pair is read: ask for that fact, and an index built over
    the whole section answers, while one built over a unit the model stopped
    reading does not.

    The prose is the corpus's own words, for the reason a sibling defect
    records: a paragraph of this module's choosing is a paragraph in this
    module's language.
    """
    words = _commonest_words(corpus, count=12)
    if not words:
        return dict(corpus)
    sentence = " ".join(words)
    number = len(corpus) + 1
    body: list[str] = []
    # Numbered, and that is a measurement: an unnumbered sentence repeated to
    # sixty thousand characters produces a unit identical to the unit beside
    # it at any chunk size, and the duplicate check fired on a defect that
    # names the model's window. The numbers run far below the fact at the end,
    # so that fact stays the only occurrence of its own value.
    while sum(len(line) + 1 for line in body) < CHARACTERS_PAST_THE_MODEL_WINDOW:
        body.append(f"{sentence} {len(body) + 1}.")
    out = dict(corpus)
    out[f"{number:02d}.md"] = (
        f"# {number} {' '.join(words[:3])}\n\n"
        + " ".join(body)
        + f" {' '.join(words[:4])}: {THE_FACT_AT_THE_END}\n"
    )
    return out


#: The fact only the end of that section carries, in the corpus's own shape: a
#: bare number nothing else in the corpus holds. Written here and not lifted,
#: because a fact lifted from the corpus is a fact the corpus states elsewhere,
#: and then finding it proves nothing about the end of this section.
THE_FACT_AT_THE_END = "8317"


def _unnumbered(title: str) -> str:
    """A heading's words without the number in front of them."""
    return re.sub(r"^[\d.]+\s*", "", title).strip()


def _split_every_section_in_two(corpus: Corpus) -> Corpus:
    """Cut each section at its middle sentence and give the halves a heading each.

    The grounds for an answer are two sentences that stood together, and now
    they stand in two units. Retrieval scores a unit against a question, so
    each half matches the question about half as well as the whole did, and a
    window that would have returned the answer returns one half of it.

    Every length, number and heading stays in order and no text is lost: the
    corpus reads as a slightly more finely divided version of itself, which is
    what makes the failure quiet.

    Two refusals shaped what it will and will not cut, and both were the
    corpus answering back. A section of one or two sentences cut in two leaves
    units whose whole text is one sentence, and a handbook repeats a few
    sentences across its documents as any handbook does: thirteen of them
    surfaced as units identical to each other, so the mutation staged a
    duplicate beside the failure it names. And the continued heading copied
    the original's words, number included, so a corpus that numbers its
    headings gained a repeated number, which the structural check reads. So a
    section is cut only when no sentence of it occurs anywhere else, and the
    continued heading carries the words without the number.
    """
    heading = re.compile(r"^(#{1,6})\s+(.*)$")
    everywhere: dict[str, int] = {}
    for text in corpus.values():
        for line in text.splitlines():
            for sentence in _sentences(line):
                everywhere[sentence] = everywhere.get(sentence, 0) + 1

    def split(text: str) -> str:
        out: list[str] = []
        level, title = "##", ""
        for line in text.splitlines():
            match = heading.match(line)
            if match:
                level, title = match.group(1), match.group(2)
                out.append(line)
                continue
            sentences = _sentences(line)
            if len(sentences) < 2 or any(everywhere[s] > 1 for s in sentences):
                out.append(line)
                continue
            middle = len(sentences) // 2
            out.append(" ".join(sentences[:middle]))
            out.append("")
            out.append(f"{level} {_unnumbered(title)}, continued")
            out.append("")
            out.append(" ".join(sentences[middle:]))
        return "\n".join(out).strip() + "\n"

    return {name: split(text) for name, text in corpus.items()}


def _sentences(line: str) -> list[str]:
    if not line.strip() or line.startswith("#"):
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", line.strip()) if s.strip()]


#: A designation no corpus this repository ships contains, checked by the
#: defect below before it writes anything: a code the corpus already states
#: elsewhere is a code that finding proves nothing about.
THE_CODE_IN_ONE_DOCUMENT = "ЩК-4471"


def _hide_a_code_in_one_document(corpus: Corpus) -> Corpus:
    """Write a designation into one document, once, and nowhere else.

    A keyword index finds it: a code is a token and the token is there. What
    the pair asks is whether the semantic half does, and the answer depends on
    how much text surrounds it. Measured on the model this proving ground
    indexes with: in a fragment of a hundred and fifty characters the code
    decides the ranking; at a thousand it is beaten by two fragments in
    twenty-six; and in four thousand fragments of ordinary encyclopaedic
    length the fragment carrying it comes back three thousand three hundred
    and eighty-seventh.

    So the defect is one sentence and the corpus decides what it stages. On a
    small corpus of short sections it stages nothing, which is why the first
    attempt at this entry recorded a blocker.
    """
    if any(THE_CODE_IN_ONE_DOCUMENT in text for text in corpus.values()):
        raise CorpusCannotCarryDefect(
            f"this corpus already states {THE_CODE_IN_ONE_DOCUMENT!r}, so a search finding it "
            "would prove nothing about the sentence this defect adds"
        )
    words = _commonest_words(corpus, count=4)
    if not words:
        return dict(corpus)
    out = dict(corpus)
    # The middle document by name, so the mutation is not on whichever
    # document a neighbouring test happens to read first.
    name = sorted(corpus)[len(corpus) // 2]
    out[name] = corpus[name].rstrip() + (
        f"\n\n{' '.join(words)}: {THE_CODE_IN_ONE_DOCUMENT}.\n"
    )
    return out


def _number_every_subsection_heading(corpus: Corpus) -> Corpus:
    """Give each subsection a number of its own, continuing the corpus's run.

    Not a defect of the documents at all, and the defect list says so: what
    this stages is a citation that names the wrong fragment, and that failure
    needs fragments a citation can name. A structural label is the heading it
    came from, so a heading with no number gives a label with no number, and
    the check that reads whether the numbers in an answer belong to the
    fragments beside it has nothing to read.

    Measured before it was written: of ninety-five labels this corpus's
    retrieval returned, twenty-one carried a number, and the coverage the
    healthy half recorded was 0.11. A pair measured against that is a pair
    measured against noise.

    Numbered straight on from the last document and never as "3.1" under
    document three, which was the first attempt: the structural check reads
    the leading integer out of a label, so every subsection of document three
    came back as another node numbered three and the mutation reported forty
    duplicated numbers. Continuing the run keeps every number distinct and
    leaves no gap in it, which is the other thing that check reads.
    """
    heading = re.compile(r"^(#{2,6})\s+(.*)$")
    next_number = len(corpus) + 1
    out: Corpus = {}
    for name in sorted(corpus):
        lines: list[str] = []
        for line in corpus[name].splitlines():
            match = heading.match(line)
            if not match or _NUMBERED_HEAD_RE.match(match.group(2)):
                lines.append(line)
                continue
            lines.append(f"{match.group(1)} {next_number} {match.group(2)}")
            next_number += 1
        out[name] = "\n".join(lines) + "\n"
    return out


#: A heading that already begins with a number needs none added.
_NUMBERED_HEAD_RE = re.compile(r"^\d")


DEFECTS: tuple[Defect, ...] = (
    Defect("flatten_headings",
           "no document carries a heading, so nothing can build a structural tree",
           ("F03",), _flatten_headings),
    Defect("duplicate_documents",
           "every document appears twice, byte for byte",
           ("F01", "F26"), _duplicate_documents),
    Defect("drop_a_numbered_document",
           "one number is missing from the middle of an otherwise sequential set",
           ("F04",), _drop_a_numbered_document,
           requires="numbers its headings, which is what the structural check reads",
           admits=_has_numbered_headings),
    Defect("repeat_a_structural_number",
           "two documents with different content carry the same number",
           ("F04",), _repeat_a_structural_number,
           requires="numbers its headings, which is what the structural check reads",
           admits=_has_numbered_headings),
    Defect("shrink_to_fragments",
           "every paragraph is cut to its first sentence",
           ("F06",), _shrink_to_fragments),
    Defect("add_a_second_language",
           "documents of a second language sit beside the first",
           ("F14", "F24"), _add_a_second_language),
    Defect("stuff_a_document_with_the_corpus_own_words",
           "one added document repeats the corpus's commonest words and says nothing",
           ("F17",), _stuff_a_document_with_the_corpus_own_words),
    Defect("repeat_a_phrase_in_every_document",
           "one ordinary sentence sits under every heading, as a template would",
           ("F41",), _repeat_a_phrase_in_every_document,
           requires="carries headings, since the phrase is placed under the first of them",
           admits=lambda corpus: any("#" in text for text in corpus.values())),
    Defect("leave_every_other_section_a_heading",
           "every other section keeps its heading and loses the text under it",
           ("F42",), _leave_every_other_section_a_heading,
           requires="carries headings, since they are all that is left of it",
           admits=lambda corpus: any("#" in text for text in corpus.values())),
    Defect("empty_the_corpus",
           "no documents at all, and every retrieval metric reporting zero",
           ("F43",), _empty_the_corpus),
    Defect("share_a_word_at_the_cap",
           "one invented word heads fifty sections, joining all fifty in one keyword bucket",
           ("F38", "F39"), _share_a_word_at_the_cap,
           requires="carries headings, since the word is placed at the head of each body",
           admits=lambda corpus: any("#" in text for text in corpus.values())),
    Defect("pad_a_section_past_the_model_window",
           "one added document is a single section longer than an embedding model reads",
           ("F13",), _pad_a_section_past_the_model_window),
    Defect("split_every_section_in_two",
           "each section is cut at its middle sentence, so grounds that stood together do not",
           ("F07",), _split_every_section_in_two,
           requires="carries headings, since each half is given one",
           admits=lambda corpus: any("#" in text for text in corpus.values())),
    Defect("hide_a_code_in_one_document",
           "one document states a designation that appears nowhere else in the corpus",
           ("F15",), _hide_a_code_in_one_document,
           requires="does not already state that designation, since finding one the corpus "
                    "states elsewhere proves nothing",
           admits=lambda corpus: not any(
               THE_CODE_IN_ONE_DOCUMENT in text for text in corpus.values())),
    Defect("number_every_subsection_heading",
           "every subsection carries a number under its document's own",
           ("F29",), _number_every_subsection_heading,
           requires="numbers its top-level headings, since a subsection's number is built "
                    "under the document's own",
           admits=_has_numbered_headings),
    Defect("cut_a_table_and_a_list",
           "one added document is mostly a table and a list, both longer than a chunk",
           ("F08",), _cut_a_table_and_a_list),
)

_BY_NAME = {d.name: d for d in DEFECTS}


def read_corpus(source: Path, limit: int = 0) -> Corpus:
    """The documents of a corpus, by file name.

    Two layouts, because this repository ships both and for a while this read
    only one. A handbook is a directory of files. A retrieval benchmark is a
    directory of directories, one per document, holding its passages; three of
    the five corpora here are laid out that way and none of them could be read
    at all, so no defect could be put into any of them.

    A nested corpus is flattened, and the name keeps both parts, so a passage
    stays identifiable and a load derives one source code for the lot. `limit`
    caps how many are read, in name order, because those corpora hold tens of
    thousands of passages and a proving ground wants a slice of one.
    """
    files = sorted(source.glob("*.md")) + sorted(source.glob("*.txt"))
    if not files:
        files = sorted(source.glob("*/*.md")) + sorted(source.glob("*/*.txt"))
        if limit:
            files = files[:limit]
        return {
            f"{path.parent.name}-{path.name}": path.read_text(encoding="utf-8")
            for path in files
        }
    if limit:
        files = files[:limit]
    return {path.name: path.read_text(encoding="utf-8") for path in files}


class CorpusCannotCarryDefect(ValueError):
    """The corpus is not the shape this defect needs.

    Raised instead of returning a corpus that looks mutated and provokes
    nothing, which is the failure this whole tool exists to stop somebody
    shipping.
    """


def mutate(corpus: Corpus, defect: str) -> Corpus:
    """The corpus with one named defect in it. Never mutates the input."""
    try:
        chosen = _BY_NAME[defect]
    except KeyError:
        raise KeyError(f"Unknown defect {defect!r}. Known: {sorted(_BY_NAME)}") from None
    if chosen.admits is not None and not chosen.admits(corpus):
        raise CorpusCannotCarryDefect(
            f"{defect!r} needs a corpus that {chosen.requires}. This one is not, so the "
            "result would look mutated and provoke nothing."
        )
    return chosen.apply(dict(corpus))


def write_corpus(corpus: Corpus, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for name, text in corpus.items():
        (destination / name).write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", nargs="?", type=Path, help="a healthy corpus directory")
    parser.add_argument("--defect", help="which defect to put in it")
    parser.add_argument("--out", type=Path, help="where to write the result")
    parser.add_argument("--list", action="store_true", help="the defects this tool can put in")
    parser.add_argument("--limit", type=int, default=0, metavar="N",
                        help="read at most N documents, in name order. For the retrieval "
                             "benchmarks, which hold tens of thousands")
    args = parser.parse_args(argv)

    if args.list or not args.source:
        width = max(len(d.name) for d in DEFECTS)
        print("Defects, and the catalogue entries each is meant to make observable:\n")
        for defect in DEFECTS:
            print(f"  {defect.name:<{width}}  {', '.join(defect.provokes)}")
            print(f"  {'':<{width}}  {defect.describes}\n")
        return 0

    if not args.defect or not args.out:
        parser.error("--defect and --out are both required when a source is given")

    corpus = read_corpus(args.source, limit=args.limit)
    if not corpus:
        parser.error(f"no documents found in {args.source}")
    try:
        broken = mutate(corpus, args.defect)
    except CorpusCannotCarryDefect as refusal:
        # An expected answer, printed as one. A stack trace here would read as a
        # crash and send a reader looking for a fault in this tool.
        print(refusal)
        return 2
    except KeyError as unknown:
        # KeyError renders its argument with quotes around it, which turns a
        # sentence into a quotation of itself.
        print(unknown.args[0])
        return 2
    write_corpus(broken, args.out)
    print(
        f"{args.source} → {args.out}: {len(corpus)} documents in, {len(broken)} out, "
        f"defect {args.defect!r}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
