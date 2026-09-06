"""The manual check tells a person where to click. Those places have to exist.

The guide's substance is prose, and prose about what should be on screen cannot
be checked by a machine. What can be checked is everything it points at: a
route the application does not serve, a query parameter it does not read, a
make target nobody defined, a module that was renamed. Each of those sends a
reader somewhere blank, and a blank page during a verification reads as a
failed verification.

Found while writing it: the corpus selection is addressed by `corpus_id` and
the first draft of the guide used `corpus`, which the page ignores. It opened,
showed a different corpus, and looked entirely plausible.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs" / "manual-verification.md"


@pytest.fixture(scope="module")
def text() -> str:
    if not GUIDE.is_file():
        pytest.fail(f"{GUIDE.relative_to(ROOT)} is missing")
    return GUIDE.read_text(encoding="utf-8")


def _links(text: str) -> list[str]:
    return re.findall(r"<http://localhost:5173([^>]*)>", text)


def test_the_guide_points_somewhere(text: str) -> None:
    """A guide with no links guards nothing, and this file would pass."""
    assert len(_links(text)) >= 4


def test_every_route_it_opens_is_one_the_application_serves(text: str) -> None:
    routes = set(re.findall(r'path="([^"]+)"',
                            (ROOT / "ui" / "src" / "App.tsx").read_text(encoding="utf-8")))
    for link in _links(text):
        path = link.split("?")[0]
        matched = any(
            path == route or (":" in route and len(path.split("/")) == len(route.split("/")))
            for route in routes
        )
        assert matched, f"the guide opens {path}, and no route serves it"


def test_every_query_parameter_it_uses_is_one_the_application_reads(text: str) -> None:
    """The one that bit: `corpus` instead of `corpus_id`. The page ignored it,
    opened on a different corpus, and looked right."""
    sources = "\n".join(p.read_text(encoding="utf-8")
                        for p in (ROOT / "ui" / "src").rglob("*.ts*")
                        if "test" not in p.parts)
    # Read as a query parameter, and not merely present somewhere in the
    # sources. Written the looser way first, and it passed on the very defect
    # it exists for: `corpus` appears in those files in other senses, so the
    # guide could pass a name the page ignores and the check would agree.
    read = set(re.findall(r"(?:searchParams|params|query)\.get\(\s*['\"]([a-zA-Z_]+)['\"]", sources))
    for link in _links(text):
        if "?" not in link:
            continue
        for pair in link.split("?", 1)[1].split("&"):
            name = pair.split("=")[0]
            assert name in read, (
                f"the guide passes {name!r}, and no page reads a parameter of that name. "
                f"Read: {sorted(read)}"
            )


def test_every_make_target_it_names_exists(text: str) -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    targets = set(re.findall(r"^([A-Za-z0-9_.-]+):", makefile, re.M))
    named = set(re.findall(r"^make ([a-z-]+)$", text, re.M))
    missing = sorted(named - targets)
    assert missing == [], f"the guide tells a reader to run targets that do not exist: {missing}"


def test_every_tool_it_names_exists(text: str) -> None:
    for module in sorted(set(re.findall(r"python3 -m (tools\.[a-z_]+|services\.[a-z_.]+)", text))):
        path = ROOT / (module.replace(".", "/") + ".py")
        assert path.is_file(), f"the guide runs {module}, and {path.relative_to(ROOT)} is not there"


def test_every_defect_it_promises_the_mutator_lists_is_one_the_mutator_has(text: str) -> None:
    """The guide quotes the six by name. A renamed defect would leave a reader
    typing a command that refuses."""
    from tools.corpus_mutate import DEFECTS

    known = {d.name for d in DEFECTS}
    quoted = set(re.findall(r"--defect ([a-z_]+)", text))
    assert quoted, "the guide names no defect, so step five instructs nobody"
    assert quoted <= known, f"the guide names defects the mutator does not have: {sorted(quoted - known)}"


def test_the_catalogue_entries_it_quotes_exist(text: str) -> None:
    from core.eval.atlas import FAILURES

    known = {f.id for f in FAILURES}
    quoted = set(re.findall(r"\*\*(F\d{2})\*\*", text)) | set(re.findall(r"\((F\d{2}(?:, F\d{2})*)\)", text))
    quoted = {i for group in quoted for i in group.split(", ")}
    assert quoted, "the guide quotes no catalogue entry"
    assert quoted <= known, f"the guide quotes entries the catalogue does not hold: {sorted(quoted - known)}"


def test_the_counts_it_states_are_the_counts_the_catalogue_has(text: str) -> None:
    """The guide says how many entries apply to each architecture. Those move
    whenever an entry is added, and a reader comparing them against the screen
    would be told the platform is wrong when the document is."""
    from core.eval.atlas import FAILURES, applicable_to
    from core.eval.rag_space import POINTS

    stated = re.search(r"(\d+) of (\d+) for the hybrid point, (\d+) for the dense\s*\n?\s*one, (\d+) for the graph one", text)
    assert stated, "the guide no longer states the counts, or states them in another shape"
    hybrid, total, dense, graph = (int(g) for g in stated.groups())
    assert total == len(FAILURES)
    assert hybrid == len(applicable_to(POINTS["hybrid"])[0])
    assert dense == len(applicable_to(POINTS["dense"])[0])
    assert graph == len(applicable_to(POINTS["graph"])[0])


def test_the_readme_points_at_it() -> None:
    """A guide nobody links to is a guide nobody reads."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "manual-verification.md" in readme
