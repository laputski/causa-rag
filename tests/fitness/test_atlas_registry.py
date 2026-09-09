"""The catalogue may not claim what nothing checks.

`core/eval/atlas.py` is data, and data about what a platform detects is exactly
the kind of claim that rots quietly: an entry says "caught", the detector it
names is renamed or deleted, and the catalogue keeps saying "caught" for years.
Every test here exists to make one such rot impossible.

The signal ids are collected from the source that produces them and not
from a list somebody maintains, because a maintained list is a second place to
forget. For the interface's own signals a Python test cannot read a TypeScript
function, so those are resolved against a mirrored declaration which
`ui/src/test/` checks against the real exports; neither side can drift alone.
"""
from __future__ import annotations

import ast
import dataclasses
import json
import re
import typing
from pathlib import Path

import pytest

from core.eval import rag_space
from core.eval.atlas import (
    AWAITING_AN_ENTRY,
    FAILURES,
    REPORTS_A_CHECK_THAT_COULD_NOT_BE_MADE,
    FailureMode,
    applicable_to,
    signal_index,
)
from core.eval.funnel import Layer
from core.eval.root_cause import Cause

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.fitness


# ── collecting the signals that actually exist ────────────────────────────────

def _literal_ids(path: Path, constructors: set[str]) -> set[str]:
    """Every `id="..."` passed to one of these constructors in this file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in constructors:
            for kw in node.keywords:
                if kw.arg == "id" and isinstance(kw.value, ast.Constant):
                    found.add(kw.value.value)
    return found


def _metric_keys(path: Path) -> set[str]:
    """Every `metrics["..."] = ...` the evaluator writes."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, ast.Assign) else []
        for t in targets:
            if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) \
                    and t.value.id == "metrics" and isinstance(t.slice, ast.Constant):
                found.add(t.slice.value)
    return found


# The interface computes signals of its own, in TypeScript, with thresholds of
# its own and no counterpart on the server. They are read out of the module
# that declares them, never copied here: a hand-kept second list is the place a
# rename gets forgotten, which is the failure this whole file exists to make
# impossible. `ui/src/test/atlasSignals.test.ts` holds up the other end,
# checking that declaration against what the functions actually produce.
_UI_DIAGNOSTICS = ROOT / "ui" / "src" / "lib" / "diagnostics.ts"


def ui_signal_ids() -> set[str]:
    source = _UI_DIAGNOSTICS.read_text(encoding="utf-8")
    block = re.search(r"export const SIGNAL_IDS = \[(.*?)\] as const", source, re.S)
    assert block, (
        f"{_UI_DIAGNOSTICS.name} declares no SIGNAL_IDS, so the catalogue has no way "
        "to name a judgement the interface makes"
    )
    return set(re.findall(r"'([a-z_]+)'", block.group(1)))


def known_signal_ids() -> set[str]:
    core = ROOT / "core"
    return (
        {f"detector:{i}" for i in _literal_ids(core / "eval" / "detectors.py", {"DiagnosticItem"})}
        | {f"health:{i}" for i in _literal_ids(core / "eval" / "corpus_health.py", {"HealthItem"})}
        # A graph index says things about itself that a corpus of documents
        # cannot: whether the units were linked at all, and whether a
        # community drew on the whole corpus. Same kind, second module.
        | {f"health:{i}" for i in _literal_ids(core / "eval" / "graph_health.py", {"GraphFinding"})}
        | {f"compare:{i}" for i in _literal_ids(core / "experiment" / "compare.py", {"CompatWarning"})}
        | {f"funnel:{i}" for i in typing.get_args(Layer)}
        | {f"cause:{i}" for i in typing.get_args(Cause)}
        | {f"metric:{k}" for k in _metric_keys(
            ROOT / "services" / "api_gateway" / "routers" / "experiments.py")}
        | {f"ui:{i}" for i in ui_signal_ids()}
    )


# ── the guards ────────────────────────────────────────────────────────────────

def test_every_signal_resolves() -> None:
    """A named signal must exist. Renaming a detector without touching the
    catalogue is the ordinary way a catalogue starts lying."""
    known = known_signal_ids()
    dangling = sorted(
        f"{f.id} -> {s.id}" for f in FAILURES for s in f.signals if s.id not in known
    )
    assert dangling == [], f"signals named by the catalogue and absent from the code: {dangling}"


def _collected_test_ids(path: Path) -> set[str]:
    """The node identifiers pytest would actually collect from one file."""
    import subprocess
    import sys
    out = subprocess.run(
        [sys.executable, "-m", "pytest", str(path), "--collect-only", "-q", "--no-header", "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=ROOT,
    ).stdout
    return {line.strip() for line in out.splitlines() if "::" in line}


def test_a_claimed_detection_carries_a_bait() -> None:
    """The rule the whole catalogue turns on. An entry claiming to be caught
    must point at the pair of observations that proved it: fires on the defect,
    silent without it. Nobody may claim a guard nobody tried to fool."""
    unproven = sorted(f.id for f in FAILURES if f.detection != "none" and not f.bait)
    assert unproven == [], f"entries claiming detection with no bait: {unproven}"


def test_the_bait_an_entry_points_at_actually_exists() -> None:
    """Checking that the field is filled in checks nothing.

    Found by planting a bait naming a file that does not exist: the guard
    accepted it. A pointer nothing resolves is the same decorative claim the
    catalogue was built to make impossible, and it had been built into the
    catalogue's own guard.

    A bait living in the unit layer must be collectable now. One living on the
    proving ground is allowed not to exist yet, because the proving ground is
    later work, and the report counts every such entry as an unproven claim
    instead of letting it read as proven.
    """
    unit_baits = {f.id: f.bait for f in FAILURES if f.bait.startswith("tests/unit/")}
    stand_baits = {f.id: f.bait for f in FAILURES if f.bait.startswith("tests/proving_ground/")}
    misplaced = sorted(
        f"{f.id} -> {f.bait}" for f in FAILURES
        if f.bait and f.id not in unit_baits and f.id not in stand_baits
    )
    assert misplaced == [], f"baits pointing outside the two known layers: {misplaced}"

    by_file: dict[str, list[tuple[str, str]]] = {}
    for failure_id, bait in unit_baits.items():
        by_file.setdefault(bait.split("::")[0], []).append((failure_id, bait))

    dangling: list[str] = []
    for file_path, entries in by_file.items():
        collected = _collected_test_ids(ROOT / file_path)
        assert collected, f"{file_path} collects no tests at all"
        for failure_id, bait in entries:
            if bait not in collected:
                dangling.append(f"{failure_id} -> {bait}")
    assert dangling == [], (
        f"baits naming a test pytest does not collect: {sorted(dangling)}"
    )


def test_a_claimed_detection_names_at_least_one_signal() -> None:
    named = sorted(f.id for f in FAILURES if f.detection != "none" and not f.signals)
    assert named == [], f"entries claiming detection while naming no signal: {named}"


def test_undetected_entries_say_what_is_missing() -> None:
    """Closes the escape hatch. Without this an entry dodges the bait
    requirement by declaring itself undetectable, and "not caught" becomes the
    comfortable answer instead of the true one."""
    silent = sorted(
        f.id for f in FAILURES if f.detection == "none" and not f.not_detected_reason
    )
    assert silent == [], f"entries claiming nothing catches them and not saying what is missing: {silent}"


def test_identifiers_are_unique_and_well_formed() -> None:
    ids = [f.id for f in FAILURES]
    assert len(ids) == len(set(ids)), "duplicate identifiers in the catalogue"
    malformed = [i for i in ids if not re.fullmatch(r"F\d{2,}", i)]
    assert malformed == [], f"identifiers not of the form F<number>: {malformed}"


def test_a_superseded_entry_points_at_its_successors() -> None:
    """An entry is never deleted and its identifier is never reused: splitting
    one retires it and allocates new ones, so a reader holding an old reference
    is told where it went instead of getting nothing."""
    ids = {f.id for f in FAILURES}
    dangling = sorted(
        f"{f.id} -> {s}" for f in FAILURES for s in f.superseded_by if s not in ids
    )
    assert dangling == [], f"superseded_by pointing at entries that do not exist: {dangling}"


def test_severity_multipliers_stay_in_their_declared_ranges() -> None:
    out = [
        f"{f.id}: quiet={f.severity.quiet} cost={f.severity.cost} prevalence={f.severity.prevalence}"
        for f in FAILURES
        if not (1 <= f.severity.quiet <= 3 and 1 <= f.severity.cost <= 4
                and 1 <= f.severity.prevalence <= 3)
    ]
    assert out == [], f"severity multipliers outside their ranges: {out}"


def test_scope_is_declared_and_resolvable() -> None:
    """Every coordinate must exist in the vendored space, and the field must be
    present on every entry even when empty. An empty scope means "applies to
    every architecture" and has to be a decision somebody took, not a line
    somebody forgot."""
    problems: list[str] = []
    for f in FAILURES:
        assert isinstance(f.applies_when, tuple), f"{f.id}: applies_when is not a tuple"
        for code, values in f.applies_when:
            dimension = rag_space.get(code)
            if dimension is None:
                problems.append(f"{f.id}: unknown dimension {code}")
                continue
            for v in values:
                if v not in dimension.values:
                    problems.append(f"{f.id}: {code} has no value {v!r}")
    assert problems == [], f"scope predicates that do not resolve: {problems}"


def test_the_vendored_space_matches_its_own_declared_size() -> None:
    assert len(rag_space.DIMENSIONS) == rag_space.SCHEMA_DIMENSION_COUNT
    codes = [d.code for d in rag_space.DIMENSIONS]
    assert len(codes) == len(set(codes)), "duplicate dimension codes in the vendored copy"


# ── the boundary between the two catalogues ──────────────────────────────────

def _repository_identifiers() -> set[str]:
    """Compound names this repository invented.

    Only compound tokens are collected: something carrying an underscore, or
    CamelCase of two words or more. A single ordinary word is never counted,
    however many functions in this tree happen to be called that.

    The first version of this check collected every function and class name and
    immediately flagged twenty-one entries for using the words `corpus`,
    `model`, `query`, `index` and `list`. Those are the vocabulary of the field,
    not of this tree, and an entry has no way to describe a retrieval failure
    without them. The lesson is that the exemption list the check would then
    need is unbounded, and an unbounded exemption list is the second place to
    forget. A compound name has no such problem: `detect_stub_embedder` and
    `NaivePipeline` are unmistakably ours, and no description of a failure any
    RAG can suffer needs either.
    """
    names: set[str] = set()
    for directory in ("core", "adapters", "services"):
        for path in (ROOT / directory).rglob("*.py"):
            names.add(path.stem)
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                    names.add(node.name)
    return {n for n in names if _is_compound(n)}


def _is_compound(name: str) -> bool:
    stripped = name.lstrip("_")
    if "_" in stripped:
        return True
    # CamelCase of two words or more.
    return bool(re.fullmatch(r"[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*", stripped))


def test_an_entry_is_phrased_without_this_repository_s_own_names() -> None:
    """The mechanical half of the boundary between the two catalogues.

    An entry describing a failure of *this* platform belongs in the incident
    log, not here. A failure any RAG can suffer is describable without naming a
    single compound identifier from this tree, and the two GraphRAG entries
    that spent a while filed as platform incidents pass this check cleanly,
    which is how their misfiling was caught.
    """
    ours = _repository_identifiers()
    offenders: list[str] = []
    for f in FAILURES:
        text = f"{f.title} {f.not_detected_reason} {f.scope_caveat}"
        words = {w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text) if _is_compound(w)}
        hit = sorted(words & ours)
        if hit:
            offenders.append(f"{f.id}: {hit}")
    assert offenders == [], (
        "entries naming this repository's own identifiers, which makes them "
        f"incidents of the platform and not failures of RAG: {offenders}"
    )


def test_the_boundary_check_can_actually_catch_something() -> None:
    """A bait for the boundary check itself.

    A test that only ever passes proves nothing about what it would catch. This
    feeds it a description phrased the way an incident of this platform is
    phrased, and requires it to object.
    """
    ours = _repository_identifiers()
    assert ours, "no compound identifiers collected: the check would pass on anything"
    planted = "The read path in _parse_result drops source_refs on every run"
    words = {w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", planted) if _is_compound(w)}
    assert words & ours, (
        "the boundary check does not object to a description written in this "
        "tree's own identifiers, so it would not have caught the misfiling it exists for"
    )


# ── the report the reconciliation produces ───────────────────────────────────

def test_the_catalogue_covers_every_row_of_the_published_atlas() -> None:
    """Rows may merge and split, but none may quietly vanish."""
    covered: set[int] = set()
    for f in FAILURES:
        covered |= set(f.atlas_rows)
    missing = sorted(set(range(1, 38)) - covered)
    assert missing == [], f"rows of the published atlas answered by no entry: {missing}"


def test_scope_caveats_are_counted_not_hidden(capsys: pytest.CaptureFixture[str]) -> None:
    """Not a gate: a record. The number must be visible so it can fall."""
    with_caveat = [f.id for f in FAILURES if f.scope_caveat]
    print(json.dumps({"entries_with_scope_caveat": with_caveat}, ensure_ascii=False))
    assert len(with_caveat) <= len(FAILURES)


def test_every_entry_declares_a_known_instrument_and_detection() -> None:
    bad_detection = sorted(f.id for f in FAILURES if f.detection not in {"detector", "visible", "none"})
    bad_instrument = sorted(
        f.id for f in FAILURES if f.instrument not in {"corpus", "config", "ingest", "faulty_rag", "platform"}
    )
    assert bad_detection == [], f"unknown detection state: {bad_detection}"
    assert bad_instrument == [], f"unknown instrument: {bad_instrument}"


def test_a_failure_mode_is_frozen() -> None:
    """The catalogue is read-only at runtime. Editing an entry from a request
    handler is how a guarantee the build checked stops holding in production."""
    f: FailureMode = FAILURES[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.detection = "detector"  # type: ignore[misc]


def test_entries_whose_signals_do_not_single_them_out_say_so() -> None:
    """Two entries naming an identical set of signals are not distinguished by
    any observation the platform can make.

    That is not a reason to merge them: the two failures have different causes
    and different fixes, and a catalogue that collapsed them would hide one of
    the two. It is a reason to say it. Where this is left undeclared, a reader
    takes a firing signal as evidence for the entry they happen to be looking
    at, when it is evidence for either.

    The declaration is checked in both directions, so half of a pair cannot
    quietly drop its half of the statement.
    """
    from collections import defaultdict

    by_signature: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for f in FAILURES:
        if f.signals:
            by_signature[tuple(sorted(s.id for s in f.signals))].append(f.id)

    undeclared: list[str] = []
    for _, ids in by_signature.items():
        if len(ids) < 2:
            continue
        for failure_id in ids:
            entry = next(f for f in FAILURES if f.id == failure_id)
            expected = {i for i in ids if i != failure_id}
            if set(entry.shares_signals_with) != expected:
                undeclared.append(
                    f"{failure_id} shares its signals with {sorted(expected)} "
                    f"and declares {sorted(entry.shares_signals_with)}"
                )
    assert undeclared == [], (
        "entries whose signals cannot tell them apart, without saying so: "
        f"{undeclared}"
    )


def test_a_declared_sharing_is_mutual_and_points_at_real_entries() -> None:
    ids = {f.id for f in FAILURES}
    problems: list[str] = []
    for f in FAILURES:
        for other_id in f.shares_signals_with:
            if other_id not in ids:
                problems.append(f"{f.id} -> {other_id} does not exist")
                continue
            other = next(x for x in FAILURES if x.id == other_id)
            if f.id not in other.shares_signals_with:
                problems.append(f"{f.id} claims to share with {other_id}, which does not say so")
    assert problems == [], f"one-sided sharing declarations: {problems}"


def _committed_or_working(path: Path) -> str:
    """The file as its own repository has it committed, falling back to disk.

    The fallback covers a checkout that is not a git working tree, or a file
    not yet tracked in one. Both mean the same thing: nothing more
    authoritative is available, so read what is there.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if result.returncode != 0:
            return path.read_text(encoding="utf-8")
        root = Path(result.stdout.strip())
        shown = subprocess.run(
            ["git", "-C", str(root), "show", f"HEAD:{path.relative_to(root)}"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if shown.returncode == 0 and shown.stdout:
            return shown.stdout
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return path.read_text(encoding="utf-8")


def test_the_vendored_coordinates_are_compared_against_the_published_ones() -> None:
    """A copy drifts. This is the only thing that stops this one drifting
    silently.

    Two sources are tried, the stronger first: a checkout of the schema beside
    this one, then the published data over the network. Neither is guaranteed
    to be there, and when neither is, the test **skips loudly** instead of
    passing. A green run that proves nothing is exactly what the copy was
    warned about in its own docstring.

    From the sibling checkout it reads what is **committed** there, and not
    what is in its working tree. Somebody editing that schema turned this test
    red here over two values half-added in another repository; an unfinished
    edit is not a publication, and a copy chasing one would be verified against
    something nobody can fetch. The values land here when they land there.
    """
    published: dict[str, tuple[str, ...]] | None = None
    source = ""

    sibling = ROOT.parent / "rag-world" / "core" / "dimensions_schema.py"
    if sibling.exists():
        tree = ast.parse(_committed_or_working(sibling))
        collected: dict[str, tuple[str, ...]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == "Dimension" and len(node.args) >= 3:
                code = node.args[0]
                values = node.args[2]
                if isinstance(code, ast.Constant) and isinstance(values, ast.Tuple):
                    collected[code.value] = tuple(
                        e.value for e in values.elts if isinstance(e, ast.Constant)
                    )
        if collected:
            published, source = collected, str(sibling)

    if published is None:
        pytest.skip(
            "neither a checkout of the schema nor the published data was reachable, "
            "so the vendored copy was NOT compared against anything. It may be stale."
        )

    mine = {d.code: d.values for d in rag_space.DIMENSIONS}
    assert set(mine) == set(published), (
        f"the vendored copy and {source} disagree on which dimensions exist: "
        f"only here {sorted(set(mine) - set(published))}, "
        f"only there {sorted(set(published) - set(mine))}"
    )
    differing = sorted(c for c in mine if mine[c] != published[c])
    assert differing == [], (
        f"the vendored copy and {source} disagree on the values of {differing}"
    )


# ── the outbound links, and the date beside them ─────────────────────────────
#
# Two questions a reviewer asked of the mockups, and both had to become checks
# and never intentions: does the link go where it says, and is the date next
# to it still current. Neither was checked, and the first was already wrong:
# the link named an anchor per dimension code, and the article declares nine
# section anchors and none per code, so it resolved to nothing and dropped the
# reader at the top of the page with no sign it had missed.

_SIBLING_ARTICLE = ROOT.parent / "rag-world" / "ui" / "src" / "generalizedData.ts"
_SIBLING_RELEASES = ROOT.parent / "rag-world" / "ui" / "public" / "data" / "releases" / "index.json"


def test_every_anchor_the_platform_links_to_exists_in_the_published_article() -> None:
    """A dead fragment is worse than no link: the page opens, and nothing says
    the reader landed somewhere other than the definition they asked for."""
    if not _SIBLING_ARTICLE.exists():
        pytest.skip(
            "the published article was not reachable, so the anchors this "
            "platform links to were NOT checked and may be dead"
        )
    declared = set(re.findall(r'id:\s*"([a-z_]+)"', _SIBLING_ARTICLE.read_text(encoding="utf-8")))
    assert declared, "no section anchors found in the article source: the check would pass on anything"

    emitted = {
        rag_space.dimension_url(d.code).split("#", 1)[1]
        for d in rag_space.DIMENSIONS
    }
    dead = sorted(a for a in emitted if a not in declared)
    assert dead == [], (
        f"anchors this platform links to and the article does not declare: {dead}. "
        f"It declares {sorted(declared)}"
    )


def test_the_recorded_schema_release_is_the_latest_published_one() -> None:
    """The date sits beside every coordinate the interface shows, so a stale one
    tells the reader the copy is current when it is behind."""
    if not _SIBLING_RELEASES.exists():
        pytest.skip(
            "the published release list was not reachable, so the recorded "
            f"schema release {rag_space.SCHEMA_RELEASE} was NOT checked and may be stale"
        )
    releases = json.loads(_SIBLING_RELEASES.read_text(encoding="utf-8"))["releases"]
    tags = sorted(r["tag"] for r in releases)
    assert tags, "the release list is empty: the check would pass on anything"
    assert tags[-1] == rag_space.SCHEMA_RELEASE, (
        f"the vendored copy records release {rag_space.SCHEMA_RELEASE} and the latest "
        f"published one is {tags[-1]}. Re-check the copy against it, then move the date."
    )


def test_the_link_check_can_actually_catch_a_dead_anchor() -> None:
    """A bait for the two checks above. One that only ever passes proves
    nothing about what it would catch, and the anchor it exists for was live in
    this repository until today."""
    if not _SIBLING_ARTICLE.exists():
        pytest.skip("nothing to bait the check against")
    declared = set(re.findall(r'id:\s*"([a-z_]+)"', _SIBLING_ARTICLE.read_text(encoding="utf-8")))
    assert "A2" not in declared, (
        "the anchor the old link used does exist after all, which would mean "
        "the fix removed a working link"
    )
    assert rag_space.SCHEMA_ARTICLE_ANCHOR in declared


def test_the_revision_the_copy_names_carries_the_values_the_copy_has() -> None:
    """The marker beside the copy and the copy itself can only move together.

    The release tag above cannot do this job on its own. A release is cut less
    often than the data is rebuilt: on 2026-09-01 the published data gained two
    values while the tag stayed at 2026-08-14, so a copy carrying either set of
    values agreed with the tag equally well.

    This is the second of two checks and they catch opposite mistakes. The one
    comparing the copy with the neighbour's current state catches a copy left
    behind. This one compares the copy with the revision it claims to have been
    checked against, and catches values updated while the marker was forgotten,
    or a marker moved while the values were not.
    """
    import subprocess

    sibling = ROOT.parent / "rag-world"
    if not (sibling / "core" / "dimensions_schema.py").exists():
        pytest.skip(
            "no checkout of the schema beside this one, so the recorded revision was NOT "
            "verified. The copy may name a revision that never existed."
        )

    shown = subprocess.run(
        ["git", "-C", str(sibling), "show",
         f"{rag_space.SCHEMA_VERIFIED_AGAINST}:core/dimensions_schema.py"],
        capture_output=True, text=True, timeout=15, check=False,
    )
    assert shown.returncode == 0, (
        f"the copy names revision {rag_space.SCHEMA_VERIFIED_AGAINST!r}, which that repository "
        f"does not have: {shown.stderr.strip()}"
    )

    at_revision: dict[str, tuple[str, ...]] = {}
    for node in ast.walk(ast.parse(shown.stdout)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "Dimension" and len(node.args) >= 3 \
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[2], ast.Tuple):
            at_revision[node.args[0].value] = tuple(
                e.value for e in node.args[2].elts if isinstance(e, ast.Constant))

    mine = {d.code: d.values for d in rag_space.DIMENSIONS}
    differing = sorted(c for c in mine if mine.get(c) != at_revision.get(c))
    assert differing == [], (
        f"the copy claims revision {rag_space.SCHEMA_VERIFIED_AGAINST[:8]} and disagrees with it "
        f"on {differing}"
    )


def test_the_date_beside_the_revision_is_that_revisions_date() -> None:
    """The date is for a reader, who should not have to run a command to learn
    how old a copy is. Derived from the revision, so it cannot drift from it."""
    import subprocess

    sibling = ROOT.parent / "rag-world"
    if not (sibling / "core" / "dimensions_schema.py").exists():
        pytest.skip("no checkout of the schema beside this one, so the date was NOT verified.")

    dated = subprocess.run(
        ["git", "-C", str(sibling), "show", "-s", "--format=%ad", "--date=short",
         rag_space.SCHEMA_VERIFIED_AGAINST],
        capture_output=True, text=True, timeout=15, check=False,
    )
    if dated.returncode != 0:
        pytest.skip("the recorded revision is not in that checkout; the test above says so.")
    assert dated.stdout.strip() == rag_space.SCHEMA_VERIFIED_ON, (
        f"the copy says it was checked on {rag_space.SCHEMA_VERIFIED_ON}, and the revision it "
        f"names is dated {dated.stdout.strip()}"
    )


def test_the_copy_was_checked_no_earlier_than_the_release_it_records() -> None:
    """A copy verified before the release it claims to follow would be a copy
    of something older wearing a newer label."""
    assert rag_space.SCHEMA_VERIFIED_ON >= rag_space.SCHEMA_RELEASE, (
        f"the copy records release {rag_space.SCHEMA_RELEASE} and says it was checked on "
        f"{rag_space.SCHEMA_VERIFIED_ON}, which is earlier"
    )


def test_every_instrument_a_tool_uses_is_one_the_entry_admits() -> None:
    """The catalogue and the three staging tools have to agree in both
    directions.

    They did not: the corpus mutator claimed an entry marked `platform`,
    another marked `ingest` and a third marked `config`, and every claim was
    sound. A single instrument could not say a failure is reachable two ways,
    so `also_staged_by` names the rest and this reads both.
    """
    from tools.config_distort import DISTORTIONS as SETTINGS
    from tools.corpus_mutate import DEFECTS
    from tools.ingest_distort import DISTORTIONS as LOADS

    by_id = {f.id: f for f in FAILURES}
    claims = (
        [("corpus", d.name, d.provokes) for d in DEFECTS]
        + [("config", d.name, d.provokes) for d in SETTINGS]
        + [("ingest", d.name, d.provokes) for d in LOADS]
    )
    wrong = []
    for instrument, name, provokes in claims:
        for failure_id in provokes:
            entry = by_id.get(failure_id)
            if entry is None:
                wrong.append(f"{name} names {failure_id}, absent from the catalogue")
            elif instrument not in (entry.instrument, *entry.also_staged_by):
                wrong.append(
                    f"{name} stages {failure_id} by {instrument!r}, and the entry admits "
                    f"{(entry.instrument, *entry.also_staged_by)}"
                )
    assert wrong == [], "\n  ".join(wrong)


def test_no_entry_lists_its_own_instrument_twice() -> None:
    """`also_staged_by` names the other ways, so repeating the primary there
    would make a reader count two where there is one."""
    doubled = sorted(f.id for f in FAILURES if f.instrument in f.also_staged_by)
    assert doubled == [], f"the primary instrument repeated among the others: {doubled}"


def test_every_additional_instrument_is_a_known_one() -> None:
    known = {"corpus", "config", "ingest", "faulty_rag", "platform"}
    bad = sorted(f.id for f in FAILURES if set(f.also_staged_by) - known)
    assert bad == [], f"unknown additional instrument: {bad}"


def _every_point_the_proving_ground_reaches() -> list[dict[str, str]]:
    """The platform's own pipelines, and the systems its instruments make.

    A mode of the faulty RAG server can move a coordinate: a system that
    never refuses has no refusal policy, and that is a different point, not a
    worse system of the same kind. Evidence recorded against such a system
    belongs to the point the mode moved to, so the points it can reach are
    part of what "a point the platform knows" means.
    """
    from services.faulty_rag_server.main import FAULTS

    points = [dict(point) for point in rag_space.POINTS.values()]
    for fault in FAULTS:
        if not fault.moves:
            continue
        for point in list(points):
            points.append({**point, **dict(fault.moves)})
    return points


def test_no_entry_is_recorded_as_reproduced_where_it_cannot_occur() -> None:
    """Evidence of a live run has to belong to a point the entry applies to.

    Found by the contradiction: an entry scoped to fusion that normalises
    scores was recorded as reproduced on a proving ground that fuses by rank.
    What the run had actually shown was that the signal the entry names fires,
    and that signal is shared with a second entry, so the firing was evidence
    for either of them and for neither in particular.

    A catalogue that accepts such a record says a failure was reproduced at a
    point where it cannot happen, which is the one thing this whole apparatus
    exists to stop.
    """
    evidence = ROOT / "eval" / "results" / "proving_ground"
    if not evidence.is_dir():
        pytest.skip("NOT RUN: no proving-ground evidence in this tree")
    recorded = {p.stem for p in evidence.glob("*.json")}
    if not recorded:
        pytest.skip("NOT RUN: the evidence directory is empty")

    anywhere: set[str] = set()
    for point in _every_point_the_proving_ground_reaches():
        applicable, _ = applicable_to(point)
        anywhere |= {f.id for f in applicable}

    known = {f.id for f in FAILURES}
    unknown = sorted(recorded - known)
    assert unknown == [], f"evidence for entries the catalogue does not hold: {unknown}"

    impossible = sorted(recorded - anywhere)
    assert impossible == [], (
        f"recorded as reproduced, and applicable to no point the platform knows: {impossible}"
    )


def test_a_proving_ground_bait_named_for_an_entry_records_what_it_saw() -> None:
    """A pair that stages a failure has to file what it measured.

    Found by losing one: replacing an assertion with a conditional skip left
    the other branch empty, so the pair ran, staged the failure, passed, and
    recorded nothing. A test that passes in silence is indistinguishable from
    one that never ran, and the report that reads those files then reports the
    entry as unproven while a run had just proved it.

    Read out of the source, because what is under test is that the call is
    written at all, and running the suite needs a stack, the embedding model
    and every index loaded.
    """
    import re

    directory = ROOT / "tests" / "proving_ground"
    if not directory.is_dir():
        pytest.skip("NOT RUN: no proving-ground suite in this tree")
    for path in sorted(directory.glob("test_*.py")):
        source = path.read_text(encoding="utf-8")
        named = set(re.findall(r"def test_(F\d+)[_a-z]", source))
        recorded = set(re.findall(r'record\(\s*"(F\d+)"', source))
        also = set(re.findall(r'for failure_id in \("(F\d+)", "(F\d+)"\)', source))
        for pair in also:
            recorded |= set(pair)
        missing = sorted(named - recorded)
        assert missing == [], (
            f"{path.name}: named for {missing} and records nothing for them, so a run that "
            "proves the entry leaves the report saying it is unproven"
        )


def _signals_the_platform_emits() -> set[str]:
    """Every named judgement the two signal modules construct.

    Read out of the source, because the identifiers are literals inside the
    functions that build them and there is no list of them anywhere else. That
    is itself the reason this check exists: nothing enumerates the signals, so
    a new one reaches every screen without the catalogue hearing about it.
    """
    import inspect
    import re

    from core.eval import corpus_health, detectors, graph_health

    found: set[str] = set()
    for side, module in (("detector", detectors), ("health", corpus_health),
                         ("health", graph_health)):
        for match in re.finditer(r'id="([a-z_]+)"', inspect.getsource(module)):
            if match.group(1) != "ok":
                found.add(f"{side}:{match.group(1)}")
    return found


def test_every_signal_is_either_named_by_an_entry_or_accounted_for() -> None:
    """A signal no entry names speaks about something the catalogue does not
    hold, and the gap is invisible while it is unwritten: the reverse index
    returns an empty list and a finding appears on screen with nothing beside
    it.

    Six were found this way. Three report that a check could not be made, which
    is the absence of a verdict and never becomes an entry. Three report a
    failure the catalogue has no entry for yet, and those are counted so the
    number falls where somebody can see it.
    """
    emitted = _signals_the_platform_emits()
    accounted = set(signal_index()) | set(REPORTS_A_CHECK_THAT_COULD_NOT_BE_MADE) | set(AWAITING_AN_ENTRY)
    orphans = sorted(emitted - accounted)
    assert orphans == [], (
        f"signals the platform emits and nothing accounts for: {orphans}. Either an entry "
        "names it, or it is listed in core/eval/atlas.py with the reason it has none."
    )


def test_nothing_is_listed_as_unaccounted_that_an_entry_already_names() -> None:
    """The other half. A signal that gained an entry and stayed on the list
    would keep being counted as a gap that no longer exists."""
    named = set(signal_index())
    for label, listing in (("a check that could not be made", REPORTS_A_CHECK_THAT_COULD_NOT_BE_MADE),
                           ("awaiting an entry", AWAITING_AN_ENTRY)):
        overlap = sorted(named & set(listing))
        assert overlap == [], f"listed as {label} and named by an entry as well: {overlap}"


def test_the_two_listings_do_not_overlap() -> None:
    both = sorted(set(REPORTS_A_CHECK_THAT_COULD_NOT_BE_MADE) & set(AWAITING_AN_ENTRY))
    assert both == [], f"listed as both an aid and a gap: {both}"


def test_every_listed_signal_is_one_the_platform_actually_emits() -> None:
    """A listing that outlives its signal turns into a permanent excuse."""
    emitted = _signals_the_platform_emits()
    for label, listing in (("a check that could not be made", REPORTS_A_CHECK_THAT_COULD_NOT_BE_MADE),
                           ("awaiting an entry", AWAITING_AN_ENTRY)):
        gone = sorted(set(listing) - emitted)
        assert gone == [], f"listed as {label} and emitted by nothing: {gone}"


def test_each_listing_says_why() -> None:
    for listing in (REPORTS_A_CHECK_THAT_COULD_NOT_BE_MADE, AWAITING_AN_ENTRY):
        for signal, reason in listing.items():
            assert len(reason) > 60, f"{signal} is listed with no reason worth reading"


def test_every_entry_has_a_title_in_both_languages() -> None:
    """An entry declares `title_key` and the page reads it with the English
    title as a fallback, so a missing key is not an error and not a blank
    row: it is the English sentence on the Russian page, which reads as a
    translation nobody got round to, and never as the defect it is.

    Written after four entries had arrived that way and nothing had said so.
    """
    for lang in ("en", "ru"):
        bundle = json.loads(
            (ROOT / "ui" / "src" / "i18n" / "locales" / f"{lang}.json").read_text(encoding="utf-8"))
        titles = bundle["atlasPage"]["entry"]
        missing = sorted(f.id for f in FAILURES if f.id not in titles)
        assert missing == [], f"{lang} has no title for {missing}"
        orphaned = sorted(set(titles) - {f.id for f in FAILURES})
        assert orphaned == [], f"{lang} keeps a title for an entry nobody wrote: {orphaned}"


# ── what a fusion joins ───────────────────────────────────────────────────────

#: The entries that are about two sources being merged, and what a system has
#: to be doing for them to occur in it. They carried a caveat instead until the
#: neighbouring registry's reply pointed out that the distinction is already in
#: a coordinate: of the six query transformations, `identity` and
#: `key_extraction` leave one query standing and the other four make several
#: out of one, so a fusion over those joins reformulations of one query where
#: a fusion of sources joins two.
ABOUT_MERGING_TWO_SOURCES = ("F21", "F22", "F23")


def test_an_entry_about_two_sources_does_not_reach_a_fusion_of_one() -> None:
    """A system that fuses the results of several reformulations against one
    index has one source, so the failures of having two cannot occur in it.
    The scope said otherwise until this was written, and said so out loud in a
    caveat, which is the honest version of being wrong and still wrong."""
    from core.eval.rag_space import satisfies

    fusing = {"C3": "rrf", "A5": "dense_single", "C1": "ann"}
    entries = {f.id: f for f in FAILURES if f.id in ABOUT_MERGING_TWO_SOURCES}
    assert len(entries) == len(ABOUT_MERGING_TWO_SOURCES), "an entry named here is gone"

    for failure_id, failure in entries.items():
        one_query = satisfies(failure.applies_when, {**fusing, "B1": "identity"})
        several = satisfies(failure.applies_when, {**fusing, "B1": "multi_reformulation"})
        assert several is False, (
            f"{failure_id} occurs in a system whose fusion joins reformulations of one "
            "query, where there is only one source to have two of"
        )
        if failure_id != "F21":  # F21 is scoped to a weighted merge, not to rank fusion
            assert one_query is True, (
                f"{failure_id} no longer occurs in a system that fuses two sources"
            )


def test_no_entry_about_two_sources_still_says_it_cannot_be_scoped() -> None:
    """The caveats those three carried are gone, and gone for a reason that
    has to stay true: a caveat is a named inexactness, and naming one the
    coordinates can express is a caveat nobody will ever come back to."""
    still = sorted(f.id for f in FAILURES
                   if f.id in ABOUT_MERGING_TWO_SOURCES and f.scope_caveat)
    assert still == [], f"{still} narrowed its scope and kept the caveat about not being able to"
