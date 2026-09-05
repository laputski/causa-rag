"""The catalogue of RAG failure modes, as data the build can check.

A prose catalogue can claim anything. This one cannot: every entry that claims
to be detected must name a signal that exists, and must point at a bait, meaning a
pair of observations proving the signal fires on the defect and stays silent
without it. `tests/fitness/test_atlas_registry.py` enforces both. An entry
claiming a detection nobody tried to disprove does not build.

Two catalogues exist and must not be mixed. This is the first: failures
possible in a RAG system, whatever built it. The second is this platform's own
incident log (`docs/DEBUGGING.md`), and the boundary is mechanical: an entry
here is phrased without a single identifier belonging to this repository. Two
GraphRAG failures sat in the second catalogue for a while, not because they
belong there but because the catalogue was assembled on a hybrid system and had
nowhere else to put them.

Scope is a coordinate, never a name. `applies_when` is a predicate over the
configuration space in `core/eval/rag_space.py`, so which entries apply to an
architecture is *derived*. Adding an architecture therefore costs no edit here;
what it costs is entries for the coordinates no existing entry requires, and
the report is what makes that gap visible before the work starts.

Where the published schema cannot express a scope, the entry says so in
`scope_caveat` instead of pretending. Three such cases were found and are
recorded below; each is also submitted to the schema owner's own residual
queue, because inventing a private dimension here would put this platform's
vocabulary at odds with the one it borrows.

Numbers here are ordering, not measurement. Severity is a product of three
multipliers, two of which are judgement, and no decision in this module or
around it rests on its magnitude.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.eval.rag_space import Point, Predicate, satisfies
from core.eval.rag_space import get as get_dimension

# Where an entry came from. `ours-N` is an incident survived on this platform
# and written up in the debugging log; `miracl` was found while measuring on a
# public benchmark; `mechanism` follows from how the parts fit together;
# `industry` is widely named in practitioner writing, which is not the same as
# demonstrated.
Origin = str

# What the platform can do about the entry today. `detector` claims a named
# signal decides it; `visible` claims the data shows it and a person concludes;
# `none` claims nothing catches it. All three are claims, and all three need
# evidence: a `none` entry staged on the proving ground carries a reverse bait,
# a recording that the failure is present and nothing fires.
Detection = str

# Which of the four means stages the failure on the proving ground.
# How a failure is put in front of the platform on the proving ground.
#
#   corpus     : a named defect in the documents (tools/corpus_mutate.py)
#   config     : a field of a run's configuration (tools/config_distort.py)
#   ingest     : a separate load of the corpus (tools/ingest_distort.py)
#   faulty_rag : an external system that distorts its own answers
#   platform   : a change to the platform's own code
#
# `ingest` was split off from `config` after both tools existed and both
# claimed entries marked `config`, leaving a reader unable to tell which one
# applied. The two are genuinely different instruments: a run's field is
# chosen when the run is launched, a load's is fixed when the index is created
# and every query afterwards inherits it.
Instrument = str


@dataclass(frozen=True)
class Severity:
    """Three multipliers, kept apart so any one of them can be argued with.

    quiet: 1 crashes, 2 returns a value that gives pause, 3 returns a plausible
    number. cost: hours, days, weeks, months. prevalence: needs a special
    build, common, near-universal. Only the first is derived from the failure's
    own description; the other two are judgement, and prevalence has no data
    under it at all.
    """

    quiet: int
    cost: int
    prevalence: int

    @property
    def total(self) -> int:
        return self.quiet * self.cost * self.prevalence


@dataclass(frozen=True)
class Signal:
    """A named judgement of the platform, and which side computes it.

    The side matters because a Python module cannot resolve a TypeScript
    function. `core` signals are resolved by reading the module that produces
    them; `ui` signals are resolved against a mirrored declaration that the
    interface's own test keeps honest. Two independent sets of signals exist
    and disagree on the same data today, which is exactly why neither may be
    left unnamed.
    """

    kind: str   # detector | health | compare | funnel | cause | metric | ui
    name: str

    @property
    def id(self) -> str:
        return f"{self.kind}:{self.name}"

    @property
    def side(self) -> str:
        return "ui" if self.kind == "ui" else "core"


@dataclass(frozen=True)
class FailureMode:
    id: str
    # Which rows of the published atlas this entry grew from. A tuple because a
    # merged entry answers several, and because an entry may in time answer
    # none of them: the register is this platform's catalogue, the document is
    # a snapshot of it on the day it was published.
    atlas_rows: tuple[int, ...]
    title: str
    # Where it arises, and the earliest stage at which a signal could exist.
    # They differ more often than not, and reading the first as the second
    # sends an engineer looking for the cause where only the symptom is.
    stage_origin: str
    stage_visible: str
    severity: Severity
    origin: Origin
    detection: Detection
    instrument: Instrument
    applies_when: Predicate
    signals: tuple[Signal, ...] = ()
    # Other instruments that can put the same failure in front of the platform.
    #
    # One value could not say this, and three tools were already contradicting
    # it: the corpus mutator stages an entry marked `platform`, another marked
    # `ingest` and a third marked `config`, and each claim is sound.
    # `instrument` stays the cheapest way and this names the rest, so a guard
    # can ask whether a tool is entitled to an entry and get an answer instead
    # of a disagreement.
    also_staged_by: tuple[Instrument, ...] = ()
    bait: str = ""
    # Required when detection is "none": what is missing, a detector or a piece
    # of knowledge nobody records. Without it an entry could dodge the bait
    # requirement by declaring itself undetectable.
    not_detected_reason: str = ""
    # What the coordinates cannot express about this entry's scope. Empty when
    # they express it fully.
    scope_caveat: str = ""
    # Entries whose signals are the same as this one's. Declared, not inferred,
    # because the consequence has to be read by a person: where two entries
    # name an identical set of signals, neither entry's signals are evidence
    # for *it* in particular. The signal says retrieval failed; which of the
    # two failures it was, nothing here can say. Left empty means the entry's
    # signals single it out, and the guard checks that claim both ways.
    shares_signals_with: tuple[str, ...] = field(default_factory=tuple)
    superseded_by: tuple[str, ...] = field(default_factory=tuple)

    @property
    def title_key(self) -> str:
        return f"atlasPage.entry.{self.id}"

    @property
    def state(self) -> str:
        """What the platform can do about this entry today, in one word.

        Four, and the difference between them is the reason the catalogue is
        worth reading. `caught`: a signal decides it and a bait proved both
        ends. `visible`: the data shows something and a person concludes.
        `unproven`: a signal is named and its bait waits on a proving ground
        that does not exist, so nobody has watched it fire. `none`: nothing
        catches it, and the entry has to say what is missing.

        Computed here because it was computed in two places for a while, once
        in a reporting tool and once in the interface, in two languages with
        nothing holding them to the same answer. That is the drift this whole
        catalogue exists to make impossible, and it had been built into the
        catalogue's own readers.
        """
        if self.detection == "none":
            return "none"
        if self.bait.startswith("tests/proving_ground/"):
            return "unproven"
        return "caught" if self.detection == "detector" else "visible"

    def applies_to(self, point: Point) -> bool | None:
        """Whether this failure is possible at a point. None when the point
        does not say. Never False, which would report a failure ruled out when
        it was merely never asked about."""
        return satisfies(self.applies_when, point)


FAILURES: tuple[FailureMode, ...] = (
    FailureMode(
        id="F01",
        atlas_rows=(1,),
        title="Re-ingestion adds instead of updating",
        stage_origin="ingest",
        stage_visible="retrieval",
        severity=Severity(3, 2, 3),
        origin="ours-2",
        detection="detector",
        instrument="ingest",
        also_staged_by=("corpus",),
        applies_when=(),
        signals=(Signal("detector", "duplicates"), Signal("health", "duplicates"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F01]",
    ),
    FailureMode(
        id="F02",
        atlas_rows=(2,),
        title="A lost source path makes chunk identifiers collide",
        stage_origin="ingest",
        stage_visible="retrieval",
        severity=Severity(3, 2, 1),
        origin="ours-6",
        detection="none",
        instrument="platform",
        applies_when=(("A2", ("fixed", "structure_aware", "late_chunking", "semantic")),),
        not_detected_reason=(
            "no signal reports a chunk-identifier collision: nothing observes how an identifier was derived"
        ),
    ),
    FailureMode(
        id="F03",
        atlas_rows=(3,),
        title="The chosen segmentation strategy silently does nothing",
        stage_origin="chunking",
        stage_visible="chunking",
        severity=Severity(3, 4, 2),
        origin="ours-8",
        detection="visible",
        instrument="platform",
        also_staged_by=("corpus",),
        applies_when=(("A2", ("fixed", "structure_aware", "late_chunking", "semantic")),),
        signals=(Signal("health", "no_structure"), Signal("health", "some_missing_path"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F03]",
    ),
    FailureMode(
        id="F04",
        atlas_rows=(4,),
        title="The source export lost or overwrote documents",
        stage_origin="data",
        stage_visible="retrieval",
        severity=Severity(3, 2, 2),
        origin="ours-10",
        detection="detector",
        instrument="corpus",
        applies_when=(),
        signals=(Signal("health", "missing_structural_numbers"), Signal("health", "duplicate_structural_numbers"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F04]",
    ),
    FailureMode(
        id="F05",
        atlas_rows=(5,),
        title="Fixed-size segmentation cuts through the middle of a thought",
        stage_origin="chunking",
        stage_visible="retrieval",
        severity=Severity(3, 3, 3),
        origin="industry",
        detection="none",
        instrument="faulty_rag",
        applies_when=(("A2", ("fixed",)),),
        not_detected_reason=(
            "nothing measures whether a chunk boundary falls inside a sentence"
        ),
    ),
    FailureMode(
        id="F06",
        atlas_rows=(6,),
        title="Chunks are too small to carry the grounds for an answer",
        stage_origin="chunking",
        stage_visible="generation",
        severity=Severity(3, 2, 2),
        origin="industry",
        detection="detector",
        instrument="corpus",
        applies_when=(("A2", ("fixed", "structure_aware", "late_chunking", "semantic")),),
        signals=(Signal("health", "too_short"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F06]",
    ),
    FailureMode(
        id="F07",
        atlas_rows=(7,),
        title="The grounds are spread across chunks and no single one wins",
        stage_origin="chunking",
        stage_visible="retrieval",
        severity=Severity(3, 3, 2),
        origin="industry",
        detection="detector",
        instrument="corpus",
        applies_when=(("A2", ("fixed", "structure_aware", "late_chunking", "semantic")),),
        signals=(Signal("cause", "chunking"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F07]",
    ),
    FailureMode(
        id="F08",
        atlas_rows=(8,),
        title="Tables and lists are destroyed by naive splitting",
        stage_origin="chunking",
        stage_visible="generation",
        severity=Severity(3, 2, 2),
        origin="industry",
        detection="none",
        instrument="corpus",
        applies_when=(("A2", ("fixed", "structure_aware", "late_chunking", "semantic")),),
        not_detected_reason=(
            "nothing inspects a chunk for a broken table or list"
        ),
    ),
    FailureMode(
        id="F09",
        atlas_rows=(9,),
        title="A stale index: the documents changed and the index did not",
        stage_origin="ingest",
        stage_visible="retrieval",
        severity=Severity(3, 3, 3),
        origin="industry",
        detection="none",
        instrument="platform",
        applies_when=(),
        not_detected_reason=(
            "no fingerprint of the source is recorded at ingest, so a reindexed corpus is indistinguishable from a fresh one"
        ),
    ),
    FailureMode(
        id="F10",
        atlas_rows=(10,),
        title="A stub embedder indexes random vectors",
        stage_origin="vectors",
        stage_visible="retrieval",
        severity=Severity(2, 3, 2),
        origin="ours-1",
        detection="detector",
        instrument="ingest",
        applies_when=(("A5", ("dense_single", "dense_multi_late_interaction")),),
        signals=(Signal("detector", "stub_embedder"), Signal("ui", "dense_score_low"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F10]",
    ),
    FailureMode(
        id="F11",
        atlas_rows=(11,),
        title="Different models at indexing time and at query time",
        stage_origin="vectors",
        stage_visible="retrieval",
        severity=Severity(3, 3, 2),
        origin="mechanism",
        detection="none",
        instrument="ingest",
        applies_when=(("A5", ("dense_single", "dense_multi_late_interaction")),),
        not_detected_reason=(
            "the embedder a corpus was indexed with is not recorded, so it cannot be compared with the one a run queries with"
        ),
    ),
    FailureMode(
        id="F12",
        atlas_rows=(12,),
        title="The model changed and the corpus was not reindexed",
        stage_origin="vectors",
        stage_visible="retrieval",
        severity=Severity(3, 3, 2),
        origin="industry",
        detection="none",
        instrument="ingest",
        applies_when=(("A5", ("dense_single", "dense_multi_late_interaction")),),
        not_detected_reason=(
            "same missing record as the entry above"
        ),
    ),
    FailureMode(
        id="F13",
        atlas_rows=(13,),
        title="A document longer than the model window is silently truncated",
        stage_origin="vectors",
        stage_visible="retrieval",
        severity=Severity(3, 2, 2),
        origin="mechanism",
        detection="none",
        instrument="corpus",
        applies_when=(("A5", ("dense_single", "dense_multi_late_interaction")),),
        not_detected_reason=(
            "nothing compares a document's length against the model's window"
        ),
    ),
    FailureMode(
        id="F14",
        atlas_rows=(14,),
        title="The model does not cover the language of the corpus",
        stage_origin="vectors",
        stage_visible="retrieval",
        severity=Severity(3, 3, 2),
        origin="miracl",
        detection="visible",
        instrument="corpus",
        applies_when=(("A5", ("dense_single", "dense_multi_late_interaction")),),
        signals=(Signal("health", "mixed_language"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F14]",
    ),
    FailureMode(
        id="F15",
        shares_signals_with=("F16",),
        atlas_rows=(15,),
        title="Semantic search misses codes, names and numbers",
        stage_origin="retrieval",
        stage_visible="retrieval",
        severity=Severity(3, 3, 3),
        origin="industry",
        detection="visible",
        instrument="corpus",
        applies_when=(("C1", ("ann",)),),
        signals=(Signal("funnel", "retrieval"), Signal("cause", "not_retrievable"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F15]",
    ),
    FailureMode(
        id="F16",
        shares_signals_with=("F15",),
        atlas_rows=(16,),
        title="Keyword search misses a paraphrase",
        stage_origin="retrieval",
        stage_visible="retrieval",
        severity=Severity(3, 2, 3),
        origin="industry",
        detection="visible",
        instrument="ingest",
        applies_when=(("C3", ("rrf", "score_normalization", "learned_fusion")),),
        signals=(Signal("funnel", "retrieval"), Signal("cause", "not_retrievable"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F16]",
        scope_caveat=(
            "the schema records only the primary representation model, so a hybrid's lexical half is invisible in it; scoped by the presence of fusion instead, which is wider than the truth"
        ),
    ),
    FailureMode(
        id="F17",
        shares_signals_with=("F21", "F40"),
        atlas_rows=(17,),
        title="Keyword search raises a document carrying a frequent word",
        stage_origin="retrieval",
        stage_visible="retrieval",
        severity=Severity(3, 2, 2),
        origin="ours-4",
        detection="detector",
        instrument="corpus",
        applies_when=(("C3", ("rrf", "score_normalization", "learned_fusion")),),
        signals=(Signal("detector", "bm25_dominance"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F17]",
        scope_caveat=(
            "the schema records only the primary representation model, so a hybrid's lexical half is invisible in it; scoped by the presence of fusion instead, which is wider than the truth"
        ),
    ),
    FailureMode(
        id="F18",
        atlas_rows=(18,),
        title="The right fragment sits below the cut-off: the selection is too narrow",
        stage_origin="retrieval",
        stage_visible="retrieval",
        severity=Severity(3, 2, 3),
        origin="industry",
        detection="detector",
        instrument="config",
        applies_when=(("D2", ("top_k", "budget_aware")),),
        signals=(Signal("cause", "ranking"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F18]",
    ),
    FailureMode(
        id="F19",
        atlas_rows=(19,),
        title="No confidence threshold: something is always returned",
        stage_origin="retrieval",
        stage_visible="generation",
        severity=Severity(3, 3, 3),
        origin="industry",
        detection="visible",
        instrument="faulty_rag",
        applies_when=(("E4", ("no_refusal",)),),
        signals=(Signal("metric", "correct_refusal"),),
        bait="tests/proving_ground/test_atlas_baits_live.py::test_bait[F19]",
    ),
    FailureMode(
        id="F20",
        atlas_rows=(20,),
        title="A metadata filter silently excludes everything",
        stage_origin="retrieval",
        stage_visible="retrieval",
        severity=Severity(3, 2, 2),
        origin="mechanism",
        detection="none",
        instrument="corpus",
        applies_when=(),
        not_detected_reason=(
            "the filters a question was asked with are not recorded on the result"
        ),
    ),
    FailureMode(
        id="F21",
        shares_signals_with=("F17", "F40"),
        atlas_rows=(21,),
        title="The halves are on incomparable scales and one of them rules the merge",
        stage_origin="fusion",
        stage_visible="retrieval",
        severity=Severity(3, 3, 2),
        origin="mechanism",
        detection="visible",
        instrument="config",
        applies_when=(("C3", ("score_normalization", "learned_fusion")),),
        signals=(Signal("detector", "bm25_dominance"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F21]",
        scope_caveat=(
            "the schema does not tell fusion of sources from fusion of reformulations of one query, so applicability to a system with a single source is unverified"
        ),
    ),
    FailureMode(
        # Found by the proving ground rather than by reading: staging a
        # dominance of one half by a setting turned out to be impossible, and
        # the load that did stage it described a failure the catalogue had no
        # entry for. A hybrid system whose second index was never built, or was
        # built and lost, answers every query from one half while every
        # configuration on file still says two.
        id="F40",
        shares_signals_with=("F17", "F21"),
        atlas_rows=(),
        title="One half of the retrieval is absent and every setting still says two",
        stage_origin="ingest",
        stage_visible="retrieval",
        severity=Severity(3, 3, 2),
        origin="mechanism",
        detection="detector",
        instrument="ingest",
        applies_when=(("C3", ("rrf", "score_normalization", "learned_fusion")),),
        signals=(Signal("detector", "bm25_dominance"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F40]",
    ),
    FailureMode(
        id="F22",
        atlas_rows=(22,),
        title="The rank-fusion constant was chosen without measuring it",
        stage_origin="fusion",
        stage_visible="retrieval",
        severity=Severity(3, 2, 3),
        origin="mechanism",
        detection="none",
        instrument="config",
        applies_when=(("C3", ("rrf",)),),
        not_detected_reason=(
            "no check reads the run history to ask whether the constant was ever varied on this "
            "question set; the value itself is recorded on every run since it became a field"
        ),
        scope_caveat=(
            "the schema does not tell fusion of sources from fusion of reformulations of one query, so applicability to a system with a single source is unverified"
        ),
    ),
    FailureMode(
        id="F23",
        atlas_rows=(23,),
        title="Both halves return the same thing: you pay for two",
        stage_origin="fusion",
        stage_visible="retrieval",
        severity=Severity(3, 2, 2),
        origin="miracl",
        detection="none",
        instrument="corpus",
        applies_when=(("C3", ("rrf", "score_normalization", "learned_fusion")),),
        not_detected_reason=(
            "nothing compares the two halves' result sets against each other"
        ),
        scope_caveat=(
            "the schema does not tell fusion of sources from fusion of reformulations of one query, so applicability to a system with a single source is unverified"
        ),
    ),
    FailureMode(
        id="F24",
        atlas_rows=(24,),
        title="The reranker does not know the language of the corpus",
        stage_origin="rerank",
        stage_visible="rerank",
        severity=Severity(3, 3, 2),
        origin="miracl",
        detection="visible",
        instrument="config",
        also_staged_by=("corpus",),
        applies_when=(("D1", ("cross_encoder", "graph_structural", "set_cover", "path_pruning")),),
        signals=(Signal("health", "mixed_language"), Signal("funnel", "rerank"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F24]",
    ),
    FailureMode(
        id="F25",
        atlas_rows=(25,),
        title="The candidate window equals the selection: reordering is possible, rescue is not",
        stage_origin="rerank",
        stage_visible="rerank",
        severity=Severity(3, 2, 3),
        origin="mechanism",
        detection="none",
        instrument="config",
        applies_when=(("D1", ("cross_encoder", "graph_structural", "set_cover", "path_pruning")),),
        not_detected_reason=(
            "no check compares the candidate window against the selection size"
        ),
    ),
    FailureMode(
        id="F26",
        atlas_rows=(26,),
        title="Near-duplicates make the ranking unstable",
        stage_origin="rerank",
        stage_visible="rerank",
        severity=Severity(3, 2, 2),
        origin="industry",
        detection="detector",
        instrument="corpus",
        applies_when=(("D1", ("cross_encoder", "graph_structural", "set_cover", "path_pruning")),),
        signals=(Signal("health", "near_duplicates"),),
        bait="tests/proving_ground/test_atlas_baits_live.py::test_bait[F26]",
    ),
    FailureMode(
        id="F27",
        atlas_rows=(27,),
        title="The price in latency is not measured",
        stage_origin="rerank",
        stage_visible="rerank",
        severity=Severity(1, 2, 3),
        origin="industry",
        detection="none",
        instrument="platform",
        applies_when=(),
        not_detected_reason=(
            "the absence of a stage trace is computed but is not surfaced as a signal"
        ),
    ),
    FailureMode(
        id="F28",
        atlas_rows=(28,),
        title="A reasoning model spends its budget and returns nothing",
        stage_origin="generation",
        stage_visible="generation",
        severity=Severity(2, 1, 2),
        origin="ours-3",
        detection="detector",
        instrument="faulty_rag",
        applies_when=(),
        signals=(Signal("detector", "empty_answers"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F28]",
    ),
    FailureMode(
        id="F29",
        atlas_rows=(29,),
        title="The right fragment was found and the answer cites another one's number",
        stage_origin="generation",
        stage_visible="citation",
        severity=Severity(3, 3, 3),
        origin="ours-11",
        detection="detector",
        instrument="faulty_rag",
        applies_when=(("E3", ("document_level", "fragment_level", "claim_level")),),
        signals=(Signal("metric", "citation_number_coverage"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F29]",
    ),
    FailureMode(
        id="F30",
        atlas_rows=(30,),
        title="An overfilled context: the middle is lost",
        stage_origin="generation",
        stage_visible="generation",
        severity=Severity(3, 3, 3),
        origin="industry",
        detection="none",
        instrument="faulty_rag",
        applies_when=(("D2", ("top_k", "budget_aware")),),
        not_detected_reason=(
            "position bias is computed on demand and is not surfaced as a signal"
        ),
    ),
    FailureMode(
        id="F31",
        atlas_rows=(31,),
        title="The model answers from its own knowledge, ignoring the context",
        stage_origin="generation",
        stage_visible="generation",
        severity=Severity(3, 3, 3),
        origin="industry",
        detection="detector",
        instrument="faulty_rag",
        applies_when=(),
        signals=(Signal("funnel", "suspected_ungrounded_answer"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F31]",
    ),
    FailureMode(
        id="F32",
        atlas_rows=(32,),
        title="Refusal is calibrated badly in both directions",
        stage_origin="generation",
        stage_visible="generation",
        severity=Severity(3, 2, 3),
        origin="industry",
        detection="detector",
        instrument="faulty_rag",
        applies_when=(("E4", ("confidence_threshold", "domain_policy")),),
        signals=(Signal("detector", "incorrect_refusals"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F32]",
    ),
    FailureMode(
        id="F33",
        atlas_rows=(33,),
        title="A metric's definition does not match its name",
        stage_origin="measurement",
        stage_visible="measurement",
        severity=Severity(3, 3, 3),
        origin="ours-5",
        detection="none",
        instrument="platform",
        applies_when=(),
        not_detected_reason=(
            "no metric declares what it computes in a form anything could check against its name"
        ),
    ),
    FailureMode(
        id="F34",
        atlas_rows=(34,),
        title="The read path loses data: the measured is blamed and the measurer is at fault",
        stage_origin="measurement",
        stage_visible="measurement",
        severity=Severity(3, 3, 2),
        origin="ours-9",
        detection="none",
        instrument="platform",
        applies_when=(),
        not_detected_reason=(
            "nothing compares what a run wrote with what a read of it returns"
        ),
    ),
    FailureMode(
        id="F35",
        atlas_rows=(35,),
        title="A metric is computed in a mode where it has no grounds",
        stage_origin="measurement",
        stage_visible="measurement",
        severity=Severity(3, 3, 2),
        origin="miracl",
        detection="none",
        instrument="platform",
        applies_when=(),
        not_detected_reason=(
            "a metric's preconditions are enforced in one place by hand and are not declared anywhere a check could read"
        ),
    ),
    FailureMode(
        id="F36",
        atlas_rows=(36,),
        title="The golden set is too small and too uniform to expose a defect",
        stage_origin="measurement",
        stage_visible="measurement",
        severity=Severity(3, 4, 3),
        origin="miracl",
        detection="none",
        instrument="platform",
        applies_when=(),
        not_detected_reason=(
            "nothing estimates the smallest difference the set can distinguish"
        ),
    ),
    FailureMode(
        id="F37",
        atlas_rows=(37,),
        title="Tuning was done on the same questions the measurement uses",
        stage_origin="measurement",
        stage_visible="measurement",
        severity=Severity(3, 3, 2),
        origin="mechanism",
        detection="none",
        instrument="platform",
        applies_when=(),
        not_detected_reason=(
            "no record ties a reported number to how many configurations were tried on the same set"
        ),
    ),
    # ── Entries that grew from no row of the published atlas ──────────────────
    #
    # The atlas was assembled on a hybrid system, so nothing in it lives on a
    # graph coordinate. These two were met on this platform and were filed for
    # a while as incidents of it, which was wrong: both are phrased without a
    # single identifier of this tree, and both are failures any keyword-linked
    # graph retriever can suffer. They were misfiled because the catalogue had
    # nowhere else to put them, which is the whole argument for scoping by
    # coordinate instead of by the architecture an atlas happened to be
    # written on.
    FailureMode(
        id="F38",
        atlas_rows=(),
        title="Linking units by shared words builds an edge count that grows with the square of the vocabulary",
        stage_origin="ingest",
        stage_visible="retrieval",
        severity=Severity(2, 3, 2),
        origin="ours-13",
        detection="none",
        instrument="corpus",
        applies_when=(("A4", ("graph", "hypergraph", "community_hierarchy")),
                      ("A8", ("extracted", "computed", "extracted_and_computed"))),
        not_detected_reason=(
            "nothing counts the edges a link step produces against the units it "
            "produced them from, so a link rule that is quadratic in a frequent "
            "word is indistinguishable from one that is not"
        ),
    ),
    FailureMode(
        id="F39",
        atlas_rows=(),
        title="Communities look like structure while the grouping follows frequent words and not meaning",
        stage_origin="ingest",
        stage_visible="retrieval",
        severity=Severity(3, 3, 2),
        origin="ours-16",
        detection="none",
        instrument="corpus",
        applies_when=(("A4", ("graph", "community_hierarchy")),
                      ("A8", ("extracted", "computed", "extracted_and_computed"))),
        not_detected_reason=(
            "a plausible modularity is reported and nothing beside it reports how "
            "many distinct documents a large community mixes, so a grouping by "
            "shared frequent words reads exactly like a grouping by meaning"
        ),
    ),
)


_BY_ID: dict[str, FailureMode] = {f.id: f for f in FAILURES}


def get(failure_id: str) -> FailureMode | None:
    return _BY_ID.get(failure_id)


def applicable_to(point: Point) -> tuple[list[FailureMode], list[FailureMode]]:
    """Split the catalogue against one architecture.

    Returns (applicable, undetermined). The second list is the point of the
    signature: a failure whose scope depends on a coordinate the point does not
    record is neither applicable nor ruled out, and collapsing it into either
    list would state something nobody established.
    """
    applicable: list[FailureMode] = []
    undetermined: list[FailureMode] = []
    for f in FAILURES:
        verdict = f.applies_to(point)
        if verdict is True:
            applicable.append(f)
        elif verdict is None:
            undetermined.append(f)
    return applicable, undetermined


def uncovered_coordinates(point: Point) -> dict[str, str]:
    """The coordinates of a point that no entry in the catalogue requires.

    This is what makes a gap visible before any work is done. A graph
    architecture today lights up every one of its own coordinates here, because
    the catalogue was assembled on a system that had none of them, and nothing
    anywhere said so.
    """
    required: set[tuple[str, str]] = set()
    for f in FAILURES:
        for code, values in f.applies_when:
            for value in values:
                required.add((code, value))
    gaps: dict[str, str] = {}
    for code, value in point.items():
        if (code, value) in required:
            continue
        # A coordinate sitting at its own default is not a gap: it is what the
        # reference point already carries, and no entry needs to speak about
        # the absence of a mechanism. Reporting those would bury the four that
        # matter under a dozen that do not.
        dimension = get_dimension(code)
        if dimension is not None and value == dimension.default:
            continue
        gaps[code] = value
    return gaps


#: Signals the platform emits that no entry names, and why.
#:
#: A signal is a named judgement, and the catalogue is what says which failure a
#: judgement is evidence for. A signal named by nothing therefore speaks about
#: something the catalogue does not hold, and until it is written down that
#: gap is invisible: the reverse index simply returns an empty list and the
#: interface shows a finding with no entry beside it.
#:
#: Two kinds live here, and telling them apart is the point. The first reports
#: that a check could not be made, which is not a failure and never will have
#: an entry. The second reports a failure the catalogue has no entry for yet,
#: and the report counts those so the number falls where somebody can see it.
REPORTS_A_CHECK_THAT_COULD_NOT_BE_MADE: dict[str, str] = {
    "detector:embedder_unverified": (
        "says the run recorded no per-half split, so whether the corpus was embedded by a "
        "working model could not be established. The absence of a verdict, and not a verdict."
    ),
    "detector:unverified_coverage": (
        "says the index was not consulted after the run, so what retrieval could have found "
        "was never checked against what it did find."
    ),
    "detector:layer_bottleneck": (
        "names the stage a run loses most at. A diagnosis pointing at where to look, which "
        "every failure of that stage shares and none of them is identified by."
    ),
}

AWAITING_AN_ENTRY: dict[str, str] = {
    "detector:header_only": (
        "a context of chunks carrying a heading and almost no body. Adjacent to the entry for "
        "chunks too small to carry an answer, and not the same failure: those hold text and "
        "too little of it, these hold a title and nothing."
    ),
    "health:header_only": (
        "the same failure read off a corpus, and never off one run's context: the whole "
        "index holds titles where it should hold text."
    ),
    "health:empty_corpus": (
        "an index holding nothing at all, which every retrieval metric reports as a total "
        "failure of the system, naming an empty corpus nowhere."
    ),
}


def signal_index() -> dict[str, list[str]]:
    """Signal id -> the failures it evidences.

    Built here and not in the modules that produce signals, so a detector
    never has to know about the catalogue that describes it. The services layer
    joins the two when it composes a response.
    """
    index: dict[str, list[str]] = {}
    for f in FAILURES:
        for s in f.signals:
            index.setdefault(s.id, []).append(f.id)
    return index
