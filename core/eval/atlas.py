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

    def state_given(self, staged: bool) -> str:
        """What the platform can do about this entry today, in one word.

        Four, and the difference between them is the reason the catalogue is
        worth reading. `caught`: a signal decides it and a bait proved both
        ends. `visible`: the data shows something and a person concludes.
        `unproven`: a signal is named, its bait runs only on a proving ground,
        and no run of it has filed what it saw, so nobody has watched it fire.
        `none`: nothing catches it, and the entry has to say what is missing.

        Computed here because it was computed in two places for a while, once
        in a reporting tool and once in the interface, in two languages with
        nothing holding them to the same answer. That is the drift this whole
        catalogue exists to make impossible, and it had been built into the
        catalogue's own readers.

        `staged` is the one part an entry cannot know about itself, and it is
        asked for and never assumed. The rule used to read the bait's path
        alone and call every proving-ground bait unproven, which was true when
        there was no proving ground and became false the day one ran: two
        entries were reproduced with their numbers on file and the interface
        went on telling a reader that nobody had proved them. A caller with no
        evidence has to say so by passing False, and gets the old answer for
        the honest reason.
        """
        if self.detection == "none":
            return "none"
        if self.bait.startswith("tests/proving_ground/") and not staged:
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
        detection="detector",
        instrument="platform",
        applies_when=(("A2", ("fixed", "structure_aware", "late_chunking", "semantic")),),
        # Written as caught by nothing on the reading that catching it would
        # need to observe how an identifier was derived. It does not: two
        # fragments carrying one identifier and different text is the
        # collision itself, and a run already returns both.
        signals=(Signal("detector", "chunk_id_collision"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F02]",
    ),
    FailureMode(
        id="F03",
        atlas_rows=(3,),
        title="The chosen segmentation strategy silently does nothing",
        stage_origin="chunking",
        stage_visible="chunking",
        severity=Severity(3, 4, 2),
        origin="ours-8",
        # Visible in the data until the strategy was asked about its own
        # output. The two health signals read an index and can say the
        # structure is absent; neither can say which setting was supposed to
        # produce it, so the reading was a person's. The third is the
        # strategy's own promise, checked at load time, and it names the
        # strategy.
        detection="detector",
        instrument="platform",
        also_staged_by=("corpus",),
        applies_when=(("A2", ("fixed", "structure_aware", "late_chunking", "semantic")),),
        signals=(
            Signal("health", "no_structure"),
            Signal("health", "some_missing_path"),
            Signal("detector", "segmentation_broke_its_promise"),
        ),
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
        detection="detector",
        instrument="platform",
        applies_when=(),
        # The fingerprint is recorded now. Every document's content hash was
        # computed at load time and thrown away; a digest over the set is
        # kept on the corpus record and carried onto every run, so two runs
        # naming one corpus can be asked whether they queried the same
        # documents. check_comparability named this gap in its own docstring
        # for as long as it existed.
        signals=(Signal("compare", "corpus_changed"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F09]",
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
        detection="detector",
        instrument="ingest",
        applies_when=(("A5", ("dense_single", "dense_multi_late_interaction")),),
        # Recorded at load time and read off what actually ran, never off the
        # configuration: its embedder field is accepted and never applied, so
        # a check reading it would compare an intention with a record.
        signals=(Signal("detector", "index_and_query_models_differ"),),
        # One observation, two entries. The model that queries differing from
        # the model that indexed is the same sentence whichever of them moved,
        # and the two are told apart by which one is meant to be right.
        shares_signals_with=("F12",),
        bait="tests/unit/test_atlas_baits.py::test_bait[F11]",
    ),
    FailureMode(
        id="F12",
        atlas_rows=(12,),
        title="The model changed and the corpus was not reindexed",
        stage_origin="vectors",
        stage_visible="retrieval",
        severity=Severity(3, 3, 2),
        origin="industry",
        detection="detector",
        instrument="ingest",
        applies_when=(("A5", ("dense_single", "dense_multi_late_interaction")),),
        # The harder half of the same comparison: the name stays and the
        # weights change, so nothing about it reaches a collection name. The
        # manifest records the version for this reason alone.
        signals=(Signal("detector", "index_and_query_models_differ"),),
        shares_signals_with=("F11",),
        bait="tests/unit/test_atlas_baits.py::test_bait[F12]",
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
        # The load and not the documents, which is what staging it settled.
        # A long document alone reaches nothing: the segmentation caps a unit
        # at its chunk size and every default is far below any model's window,
        # so the corpus half was measured to provoke exactly nothing. What
        # reaches the window is a load asked for units larger than it, and
        # then a section longer than the window is one unit whose end no
        # vector holds. Both halves are needed and only one of them decides.
        instrument="ingest",
        also_staged_by=("corpus",),
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
        applies_when=(("C3", ("rrf", "score_normalization", "learned_fusion")),
                      # Two of the six query transformations leave one query
                      # standing; the other four make several out of one, and a
                      # fusion over those joins reformulations of one
                      # query where a fusion of sources joins two.                      #
                      # The trade this makes, since a scope is only ever the
                      # nearest expressible thing: a system that fuses two
                      # sources *and* asks several queries is now ruled out
                      # where it used to be included. No record of the
                      # seventy-six exhibits that, and two exhibit the error it
                      # replaces.
                      ("B1", ("identity", "key_extraction")),
                      ),
        signals=(Signal("funnel", "retrieval"), Signal("cause", "not_retrievable"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F16]",
        scope_caveat=(
            "the schema records only the primary representation model, so a hybrid's lexical half is invisible in it; scoped by the presence of fusion instead, which is wider than the truth"
        ),
    ),
    FailureMode(
        id="F17",
        atlas_rows=(17,),
        title="Keyword search raises a document carrying a frequent word",
        stage_origin="retrieval",
        stage_visible="retrieval",
        severity=Severity(3, 2, 2),
        origin="ours-4",
        # Named a signal until staging it on a live index said otherwise, and
        # the correction is the point of staging anything. A document of the
        # corpus's own frequent words enters the keyword half's window and not
        # the semantic half's, on five questions of nineteen, and the signal
        # this entry named stays silent throughout. That signal counts the
        # share of a whole context contributed by one half, and reports a half
        # that has stopped contributing; this failure is one document raised
        # by one half, which moves that share by a fragment.
        #
        # The fixture that stood here made every fragment of every context
        # come from the keyword half alone. That is a whole context ruled by
        # one half, which is what the two entries below are about, so the
        # fixture had been proving a different failure under this name.
        detection="none",
        instrument="corpus",
        applies_when=(("C3", ("rrf", "score_normalization", "learned_fusion")),
                      # Two of the six query transformations leave one query
                      # standing; the other four make several out of one, and a
                      # fusion over those joins reformulations of one
                      # query where a fusion of sources joins two.
                      ("B1", ("identity", "key_extraction")),
                      ),
        not_detected_reason=(
            "nothing reads which half raised one fragment; the only judgement about the "
            "halves counts the share of a whole context that came from one of them, and one "
            "document moves that share by a fragment"
        ),
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
        shares_signals_with=("F40",),
        atlas_rows=(21,),
        title="The halves are on incomparable scales and one of them rules the merge",
        stage_origin="fusion",
        stage_visible="retrieval",
        severity=Severity(3, 3, 2),
        origin="mechanism",
        detection="visible",
        instrument="config",
        applies_when=(("C3", ("score_normalization", "learned_fusion")),
                      # Two of the six query transformations leave one query
                      # standing; the other four make several out of one, and a
                      # fusion over those joins reformulations of one
                      # query where a fusion of sources joins two. That distinction used to be a caveat here,
                      # saying the schema could not express it. It can: the
                      # information is in the coordinate beside this one, and
                      # the two registry records that fuse reformulations are
                      # exactly the two carrying `multi_reformulation`.
                      ("B1", ("identity", "key_extraction")),
                      ),
        signals=(Signal("detector", "bm25_dominance"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F21]",
    ),
    FailureMode(
        # Found by the proving ground rather than by reading: staging a
        # dominance of one half by a setting turned out to be impossible, and
        # the load that did stage it described a failure the catalogue had no
        # entry for. A hybrid system whose second index was never built, or was
        # built and lost, answers every query from one half while every
        # configuration on file still says two.
        id="F40",
        shares_signals_with=("F21",),
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
        detection="detector",
        instrument="config",
        applies_when=(("C3", ("rrf",)),
                      # Two of the six query transformations leave one query
                      # standing; the other four make several out of one, and a
                      # fusion over those joins reformulations of one
                      # query where a fusion of sources joins two. That distinction used to be a caveat here,
                      # saying the schema could not express it. It can: the
                      # information is in the coordinate beside this one, and
                      # the two registry records that fuse reformulations are
                      # exactly the two carrying `multi_reformulation`.
                      ("B1", ("identity", "key_extraction")),
                      ),
        # A check reads the run history now: every rank-fusion run on one
        # question set carrying the same constant is a constant nobody has
        # measured here, whatever the paper it came from says.
        signals=(Signal("detector", "fusion_constant_never_varied"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F22]",
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
        applies_when=(("C3", ("rrf", "score_normalization", "learned_fusion")),
                      # Two of the six query transformations leave one query
                      # standing; the other four make several out of one, and a
                      # fusion over those joins reformulations of one
                      # query where a fusion of sources joins two. That distinction used to be a caveat here,
                      # saying the schema could not express it. It can: the
                      # information is in the coordinate beside this one, and
                      # the two registry records that fuse reformulations are
                      # exactly the two carrying `multi_reformulation`.
                      ("B1", ("identity", "key_extraction")),
                      ),
        not_detected_reason=(
            "nothing compares the two halves' result sets against each other"
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
        detection="detector",
        instrument="platform",
        applies_when=(),
        # The absence of a whole trace was computed and never surfaced, and
        # the narrower thing was not computed at all: a trace that is present
        # and silent about a stage that ran. That is the state the price
        # hides in, and it is now named.
        signals=(Signal("detector", "unmeasured_stage_cost"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F27]",
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
        # The server prints the wrong number, and the corpus decides whether
        # there is a number to print. Staging it settled that: only the
        # top-level heading of each document carries one, retrieval returns
        # subsections, and twenty-one of ninety-five returned labels had a
        # number at all, so a server moving every citation scored like its
        # control. With every subsection numbered the same server drops the
        # coverage by half. Both halves are needed and only one of them lies.
        instrument="faulty_rag",
        also_staged_by=("corpus",),
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
            "position bias is computed on demand and is not surfaced as a signal. A "
            "proving-ground pair filled the context past what was asked for and answered "
            "from its edges: the only thing that spoke was the funnel bottleneck, naming "
            "the generation layer on ten of seventeen answerable questions, which every "
            "failure of that layer raises and none of them is identified by"
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
        detection="detector",
        instrument="platform",
        applies_when=(),
        # Nothing decides that a name is right. What is reported is the state
        # in which the question cannot be asked at all, which is the state
        # every such drift hides in: a number whose definition is declared
        # nowhere. The declarations live in core/eval/metric_definitions.py.
        signals=(Signal("detector", "undeclared_metric"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F33]",
    ),
    FailureMode(
        id="F34",
        atlas_rows=(34,),
        title="The read path loses data: the measured is blamed and the measurer is at fault",
        stage_origin="measurement",
        stage_visible="measurement",
        severity=Severity(3, 3, 2),
        origin="ours-9",
        detection="detector",
        instrument="platform",
        applies_when=(),
        # Something compares them now, and within one document: every
        # aggregate is the mean of the values its own questions carry, so a
        # question lost on the way out moves the aggregate and nothing else.
        signals=(Signal("detector", "aggregate_disagrees"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F34]",
    ),
    FailureMode(
        id="F35",
        atlas_rows=(35,),
        title="A metric is computed in a mode where it has no grounds",
        stage_origin="measurement",
        stage_visible="measurement",
        severity=Severity(3, 3, 2),
        origin="miracl",
        detection="detector",
        instrument="platform",
        applies_when=(),
        # They were enforced in one place by hand, inside the evaluator,
        # where a reader could not see them and nothing could read them.
        # Declared beside each metric now, in core/eval/metric_definitions.py,
        # and a detector reads them.
        signals=(Signal("detector", "metric_without_grounds"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F35]",
    ),
    FailureMode(
        id="F36",
        atlas_rows=(36,),
        title="The golden set is too small and too uniform to expose a defect",
        stage_origin="measurement",
        stage_visible="measurement",
        severity=Severity(3, 4, 3),
        origin="miracl",
        detection="detector",
        instrument="platform",
        applies_when=(),
        # Something estimates it now, and it is asked where somebody is
        # actually drawing a conclusion: a comparison. Asking it of a single
        # run was tried first and rejected by measurement, because every one
        # of the forty-three runs stored here falls at least three times
        # short of the difference this platform calls a regression, and a
        # signal that fires on everything distinguishes nothing.
        signals=(Signal("compare", "below_the_sets_resolution"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F36]",
    ),
    FailureMode(
        id="F37",
        atlas_rows=(37,),
        title="Tuning was done on the same questions the measurement uses",
        stage_origin="measurement",
        stage_visible="measurement",
        severity=Severity(3, 3, 2),
        origin="mechanism",
        detection="detector",
        instrument="platform",
        applies_when=(),
        # The record exists now, computed from the run store when a run is
        # read: how many configurations were tried on this question set
        # before it. A figure chosen after a search is partly the search,
        # and how much of it cannot be recovered from the figure.
        signals=(Signal("detector", "tuned_on_the_measurement_set"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F37]",
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
            "produced them from. A check for it was written and withdrawn: what a "
            "single graph can show is its consequence and never the growth itself: a unit "
            "reaching a large share of the corpus in one step, and the healthy "
            "proving-ground graph sits close enough to any such ratio to make the "
            "number a matter of taste"
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
            "a plausible modularity is reported and nothing beside it separates a "
            "grouping by meaning from a grouping by a shared word. Counting the "
            "documents a large community draws on was tried and refuted by the "
            "control: on a healthy proving-ground graph the largest community holds "
            "thirty-five units from thirty-five of forty documents, because a corpus "
            "of procedures on one subject genuinely groups across its documents"
        ),
    ),
    FailureMode(
        id="F42",
        atlas_rows=(),
        title="Fragments carry a heading and no body",
        stage_origin="chunking",
        stage_visible="generation",
        severity=Severity(3, 3, 2),
        origin="mechanism",
        detection="detector",
        instrument="corpus",
        applies_when=(("A2", ("fixed", "structure_aware", "late_chunking", "semantic")),),
        # Adjacent to the entry for fragments too small to carry the grounds
        # and not the same one: there the fragment says too little, here it
        # says the title of what it should have said. Retrieval looks right,
        # because a title matches a question about its subject better than
        # most prose does, and the answer has nothing under it.
        #
        # Two signals, and the difference between them is which question was
        # asked. One reads a run's context and says this run was answered off
        # titles; the other reads the whole index and says the corpus is made
        # of them. A corpus can be sound and one context still be titles.
        signals=(Signal("detector", "header_only"), Signal("health", "header_only")),
        bait="tests/unit/test_atlas_baits.py::test_bait[F42]",
    ),
    FailureMode(
        id="F43",
        atlas_rows=(),
        title="The index holds nothing and every metric reports a total failure",
        stage_origin="ingest",
        stage_visible="retrieval",
        severity=Severity(1, 4, 2),
        origin="mechanism",
        detection="detector",
        instrument="corpus",
        applies_when=(),
        # The quietest thing here is not the emptiness, which anybody would
        # notice on the corpus page; it is what the numbers do with it. Every
        # retrieval metric comes back at zero, which is the same shape a
        # catastrophically bad retriever produces, and a reader comparing two
        # configurations against an empty index compares nothing twice.
        signals=(Signal("health", "empty_corpus"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F43]",
    ),
    FailureMode(
        id="F44",
        atlas_rows=(),
        title="The segmentation a run declares is not the one its index was built with",
        stage_origin="chunking",
        stage_visible="retrieval",
        severity=Severity(3, 3, 2),
        origin="mechanism",
        detection="detector",
        instrument="config",
        applies_when=(("A2", ("fixed", "structure_aware", "late_chunking", "semantic")),),
        # A corpus is cut when it is loaded, so a run naming a segmentation
        # selects nothing: it reads whatever the cut produced. The name is
        # recorded all the same, on the run and on every comparison built from
        # it, so two runs differing only in that name are two runs of one
        # strategy while the screen says they are two, and the answer to "what
        # does this segmentation cost" comes back as "nothing".
        #
        # Distinct from the entry about a segmentation breaking its own
        # promise, and by the rule that separates them: the fixes differ. That
        # one is fixed in the strategy, which produced something other than
        # what its name commits it to. This one is fixed by loading the corpus
        # the run wants, or by recording what was really read.
        signals=(Signal("detector", "named_and_indexed_segmentation_differ"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F44]",
    ),
    FailureMode(
        id="F41",
        atlas_rows=(),
        title="The guard against a runaway link step leaves a graph with no edges at all",
        stage_origin="ingest",
        stage_visible="retrieval",
        severity=Severity(3, 3, 2),
        origin="ours-17",
        detection="detector",
        instrument="corpus",
        applies_when=(("A4", ("graph", "hypergraph", "community_hierarchy")),
                      ("A8", ("extracted", "computed", "extracted_and_computed"))),
        # Found on the proving ground while trying to stage the entry above,
        # and it is that entry's own prevention misfiring. Units are linked by
        # the keywords they share, and a keyword occurring in more units than
        # a frequency cap allows is excluded before linking, precisely because
        # one such keyword contributes N×(N−1)/2 edges by itself. One ordinary
        # sentence repeated under every heading puts every unit's keywords
        # over that cap at once, and the link step then produces nothing.
        #
        # The graph still exists. Its nodes are counted, its communities are
        # counted, and its modularity is perfect, because every node is its
        # own community. A graph retrieval walks to neighbours that are not
        # there and returns what a plain search would have returned.
        signals=(Signal("health", "graph_has_no_edges"),),
        bait="tests/unit/test_atlas_baits.py::test_bait[F41]",
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

AWAITING_AN_ENTRY: dict[str, str] = {}


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
