"""Put one known defect into the loading of a corpus, and nothing else.

The third of the three instruments, beside `tools/corpus_mutate.py`, which
breaks the documents, and `tools/config_distort.py`, which breaks the settings
of a run. This one breaks neither: the documents are healthy and the run is
healthy, and what carries the defect is the index they meet in.

Those defects cannot be staged any other way. An analyser is chosen when an
index is created and every query afterwards inherits it; the model a corpus was
embedded with is fixed at load time and no later setting revisits it. A run
configuration can say nothing about either.

**A plan is a pair.** Each distortion emits the healthy load as well as the
broken one, into two corpus namespaces, because a broken index measured against
nothing measures nothing. The healthy proving-ground corpora are loaded by
their own seed under one strategy, and several of these defects need another,
so a pair carrying only its broken half would be compared against a corpus that
differs from it in more than the defect.

**Nothing here runs anything.** It prints the commands, and they are the same
commands a person would type. A tool that both described and performed the load
would be free to describe one thing and perform another, which is the shape of
half the failures in the catalogue.

    python3 -m tools.ingest_distort --list
    python3 -m tools.ingest_distort --plan index_under_the_other_analyser --corpus base-ru
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field

#: The strategy the healthy proving ground is loaded under.
DEFAULT_STRATEGY = "structure_aware"
REALM_ID = "proving-ground"


@dataclass(frozen=True)
class Step:
    """One command of a plan, and why it is there."""

    #: The argv, as a person would type it. `env` names variables that must be
    #: set for this one command, which is how the embedder is chosen: it is
    #: read from the environment and has no flag of its own.
    #:
    #: Empty for a step that is not a command. One defect lives between the
    #: load and the gateway that answers, so half of it is an instruction and
    #: not an invocation; writing that instruction as a shell comment would
    #: have made a plan that looks runnable end to end and is not.
    argv: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    note: str = ""

    @property
    def is_command(self) -> bool:
        return bool(self.argv)

    def as_shell(self) -> str:
        if not self.argv:
            raise ValueError("this step is an instruction and not a command; read `note`")
        prefix = "".join(f"{name}={value} " for name, value in self.env)
        return prefix + " ".join(self.argv)


@dataclass(frozen=True)
class Plan:
    """A healthy load and a broken one, and the names they land under."""

    control_corpus_id: str
    distorted_corpus_id: str
    control: tuple[Step, ...]
    distorted: tuple[Step, ...]


@dataclass(frozen=True)
class IngestDistortion:
    """One named way to break the loading of a corpus."""

    name: str
    #: What the index looks like afterwards, in the words a reader would use.
    describes: str
    #: The catalogue entries this is meant to make observable.
    provokes: tuple[str, ...]
    build: Callable[[str, str, str, str], Plan]
    #: What the healthy load must be for this defect to exist, and an empty
    #: string when anything will do.
    requires: str = ""
    #: Whether this load can carry the defect, given corpus, language and
    #: strategy. A predicate and not a phrase read out of `requires`: deciding
    #: by searching prose for a word is a coupling nothing enforces, and the
    #: first rewording of the sentence would silently admit everything.
    admits: Callable[[str, str, str], bool] | None = None
    #: What a run staging this can prove. "signal" means a named judgement of
    #: the platform is expected to redden; "silence" means the defect is
    #: present and the platform has nothing that speaks about it, and the
    #: evidence sought is the silence itself.
    proves: str = "signal"
    #: Why only silence, required whenever `proves` says so.
    proves_note: str = ""
    #: Languages this defect needs the corpus to be in, empty when any will do.
    languages: tuple[str, ...] = field(default_factory=tuple)


def _ingest(path: str, corpus_id: str, language: str, strategy: str = DEFAULT_STRATEGY,
            chunk_size: int | None = None, real_model: bool = True, note: str = "") -> Step:
    argv = ["python3", "-m", "services.ingestion.cli", "ingest", path,
            "--strategy", strategy, "--corpus-id", corpus_id,
            "--language", language, "--realm-id", REALM_ID]
    if chunk_size is not None:
        argv += ["--chunk-size", str(chunk_size)]
    return Step(tuple(argv),
                env=(("USE_REAL_BGE_M3", "true" if real_model else "false"),),
                note=note)


def _reingest_with_another_chunk_size(path: str, corpus_id: str, language: str, strategy: str) -> Plan:
    """Load the same documents twice under two chunk sizes.

    Neither store removes anything: both write by chunk identifier, and the
    identifier of a fixed-window chunk is derived from its boundaries. Measured
    on one document of the proving ground: five hundred characters gave one
    chunk and three hundred gave two, sharing no identifier, so the collection
    afterwards holds all three. The name of the collection carries the strategy
    and the model and not the chunk size, so both loads land in the same one.

    Under the structural strategy the identifier is the section path, which a
    chunk size does not move: the same measurement gave three chunks against
    three, all identifiers shared. That is why this refuses there.
    """
    broken = f"{corpus_id}-{'reingested'}"
    return Plan(
        control_corpus_id=f"{corpus_id}-fixed",
        distorted_corpus_id=broken,
        control=(_ingest(path, f"{corpus_id}-fixed", language, strategy, chunk_size=500,
                         note="one load, one chunking"),),
        distorted=(
            _ingest(path, broken, language, strategy, chunk_size=500,
                    note="the first load, exactly like the control"),
            _ingest(path, broken, language, strategy, chunk_size=300,
                    note="the second: new identifiers, and the first load's chunks stay"),
        ),
    )


def _index_with_the_stub_embedder(path: str, corpus_id: str, language: str, strategy: str) -> Plan:
    """Load with the stub, which returns a deterministic vector per text and no
    relation between two of them.

    Measured: two texts of the same meaning in two languages came out at a
    cosine of minus nought point zero one, so the semantic half of retrieval
    becomes noise while remaining perfectly well formed. The vector has the
    same width as the real model's, so nothing downstream refuses it.
    """
    broken = f"{corpus_id}-stubbed"
    return Plan(
        control_corpus_id=corpus_id,
        distorted_corpus_id=broken,
        control=(_ingest(path, corpus_id, language, strategy, note="the real model"),),
        distorted=(_ingest(path, broken, language, strategy, real_model=False,
                           note="the stub: well-formed vectors that mean nothing"),),
    )


def _index_and_query_with_different_models(path: str, corpus_id: str, language: str, strategy: str) -> Plan:
    """Load with one model and query with another.

    The load is the easy half; the second half is a property of the gateway
    that answers, so the plan says it in words. Both models write vectors of the
    same width into a collection whose name carries the model's identifier and
    not the model, so nothing refuses the mixture and no record of it is kept.

    Two catalogue entries share this arrangement and are told apart only by
    their history: one is two models used at once by mistake, the other is one
    model replaced after the corpus was loaded. The index is identical in both
    cases, and the platform records neither the model a corpus was loaded with
    nor when it changed. That absence is the reason both entries are
    undetectable, and it is what this pair is meant to demonstrate.
    """
    broken = f"{corpus_id}-mismatched"
    return Plan(
        control_corpus_id=corpus_id,
        distorted_corpus_id=broken,
        control=(_ingest(path, corpus_id, language, strategy,
                         note="loaded and queried by the same model"),),
        distorted=(
            _ingest(path, broken, language, strategy, real_model=False,
                    note="loaded by the stub"),
            Step(note=("then run the gateway with USE_REAL_BGE_M3=true and query this corpus: "
                       "the mismatch lives between the load and the service, so this half is an "
                       "instruction and not a command")),
        ),
    )


def _index_under_the_other_analyser(path: str, corpus_id: str, language: str, strategy: str) -> Plan:
    """Create the lexical index under an analyser for another language.

    The analyser stems and tokenises by one language's rules, and it is chosen
    when the index is created; every query afterwards inherits it. Russian text
    under the English analyser keeps every word form apart, so a question
    phrased in another case or number stops matching the sentence that answers
    it, which is a keyword search missing a paraphrase.

    A fresh namespace on purpose. Loading into the existing index would be
    refused by the platform's own guard, which exists precisely so that a
    mismatch cannot arrive quietly.
    """
    other = "en" if not language.startswith("en") else "ru_be"
    broken = f"{corpus_id}-{other}-analyser"
    return Plan(
        control_corpus_id=corpus_id,
        distorted_corpus_id=broken,
        control=(_ingest(path, corpus_id, language, strategy,
                         note=f"the analyser of its own language, {language}"),),
        distorted=(_ingest(path, broken, other, strategy,
                           note=f"the analyser of another language, {other}"),),
    )


DISTORTIONS: tuple[IngestDistortion, ...] = (
    IngestDistortion(
        "reingest_with_another_chunk_size",
        "the collection holds two chunkings of the same text, the older one never removed",
        ("F01",), _reingest_with_another_chunk_size,
        requires=(
            "uses the fixed-window strategy, since a structural chunk's identifier is its "
            "section path and a chunk size does not move it"
        ),
        admits=lambda corpus_id, language, strategy: strategy == "fixed",
    ),
    IngestDistortion(
        "index_with_the_stub_embedder",
        "the semantic half of the index is well-formed vectors that mean nothing",
        ("F10",), _index_with_the_stub_embedder,
    ),
    IngestDistortion(
        "index_and_query_with_different_models",
        "the corpus is embedded by one model and questioned by another",
        ("F11", "F12"), _index_and_query_with_different_models,
        proves="silence",
        proves_note=(
            "nothing records the model a corpus was loaded with, so the two models cannot be "
            "compared and no signal can speak; the evidence sought is that every signal stays "
            "quiet while retrieval is demonstrably ruined"
        ),
    ),
    IngestDistortion(
        "index_under_the_other_analyser",
        "the lexical index stems the text by another language's rules",
        ("F16",), _index_under_the_other_analyser,
    ),
)

_BY_NAME = {d.name: d for d in DISTORTIONS}


class LoadCannotCarryDistortion(ValueError):
    """The healthy load is not the shape this distortion needs."""


class PlanDoesNotSeparateItsHalves(AssertionError):
    """The two halves of a plan would land in one namespace.

    A bug in this module, caught here so it cannot reach a proving ground,
    where the broken load would overwrite the control and the pair would
    compare a corpus against itself.
    """


def plan(
    distortion: str, path: str, corpus_id: str, language: str, strategy: str = DEFAULT_STRATEGY,
) -> Plan:
    """The pair of loads that stages one named defect. Nothing is executed."""
    try:
        chosen = _BY_NAME[distortion]
    except KeyError:
        raise KeyError(f"Unknown distortion {distortion!r}. Known: {sorted(_BY_NAME)}") from None
    if chosen.admits is not None and not chosen.admits(corpus_id, language, strategy):
        raise LoadCannotCarryDistortion(
            f"{distortion!r} needs a load that {chosen.requires}. This one is not, so the result "
            "would look distorted and provoke nothing."
        )
    built = chosen.build(path, corpus_id, language, strategy)
    if built.control_corpus_id == built.distorted_corpus_id:
        raise PlanDoesNotSeparateItsHalves(
            f"{distortion!r} would load both halves into {built.control_corpus_id!r}"
        )
    return built


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--list", action="store_true", help="the distortions this tool can plan")
    parser.add_argument("--plan", help="print the two loads of one distortion")
    parser.add_argument("--corpus", default="base-ru", help="the corpus the pair is built from")
    parser.add_argument("--path", default="", help="the documents, defaulting to the proving ground's")
    parser.add_argument("--language", default="", help="the corpus's own analyser language")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    args = parser.parse_args(argv)

    if args.plan:
        from tools.seed_proving_ground import CORPORA, GROUND_DIR

        known = {c.corpus_id: c for c in CORPORA}
        corpus = known.get(args.corpus)
        path = args.path or (str(GROUND_DIR / corpus.directory) if corpus else "")
        language = args.language or (corpus.language if corpus else "")
        if not path or not language:
            parser.error(f"unknown corpus {args.corpus!r}; pass --path and --language")
        try:
            built = plan(args.plan, path, args.corpus, language, args.strategy)
        except (LoadCannotCarryDistortion, KeyError) as refused:
            # Printed, and not raised: a person asking for a plan that cannot
            # be staged needs the reason, and a traceback buries it under the
            # frames of this file.
            print(f"refused: {refused.args[0] if refused.args else refused}")
            return 1
        chosen = _BY_NAME[args.plan]
        print(f"{chosen.name}: {chosen.describes}")
        print(f"catalogue: {', '.join(chosen.provokes)}   proves: {chosen.proves}")
        if chosen.proves_note:
            print(f"  {chosen.proves_note}")
        for label, steps, namespace in (
            ("control", built.control, built.control_corpus_id),
            ("distorted", built.distorted, built.distorted_corpus_id),
        ):
            print(f"\n  {label}, corpus {namespace}")
            for step in steps:
                if step.is_command:
                    print(f"    {step.as_shell()}")
                    if step.note:
                        print(f"      # {step.note}")
                else:
                    print(f"    (not a command) {step.note}")
        return 0

    width = max(len(d.name) for d in DISTORTIONS)
    print("Ingest distortions, and the catalogue entries each is meant to make observable:\n")
    for distortion in DISTORTIONS:
        print(f"  {distortion.name:<{width}}  {', '.join(distortion.provokes)}  ({distortion.proves})")
        print(f"  {'':<{width}}  {distortion.describes}")
        if distortion.requires:
            print(f"  {'':<{width}}  needs a load that {distortion.requires}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
