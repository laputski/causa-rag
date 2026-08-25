"""The judge frameworks stay optional, and the code stays able to run without them.

`deepeval`, `ragas` and `trulens` resolve into a constraint problem large enough
that installing them is the slow half of a first install: over twenty minutes,
with the solver rather than the network as the bottleneck. Everyone who clones
the repository paid that, including the majority who never run a judge.

They now live in their own `judges` extra. That split only holds if two things
stay true, and neither is obvious enough to survive on memory:

  * nothing outside the judge runners imports them, and the runners import them
    lazily, so a platform installed without the extra still starts;
  * the extras themselves stay separated, so a later edit does not quietly fold
    them back into `integration`.

Both are checked here rather than left to a reviewer noticing.
"""
from __future__ import annotations

import ast
import pathlib
import tomllib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

JUDGE_PACKAGES = {"deepeval", "ragas", "trulens", "trulens_eval"}

# The modules that are allowed to reach a judge at all. Each one is a runner
# whose whole purpose is to drive that framework.
JUDGE_RUNNERS = {
    "eval/deepeval_runner.py",
    "eval/ragas_runner.py",
    "eval/trulens_runner.py",
}


def _extras() -> dict[str, list[str]]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)["project"]["optional-dependencies"]


def _dist_name(requirement: str) -> str:
    """The distribution name out of a requirement string, lowercased."""
    for sep in (">=", "==", "<=", "~=", ">", "<", "[", ";"):
        requirement = requirement.split(sep)[0]
    return requirement.strip().lower()


def _module_level_imports(path: pathlib.Path) -> set[str]:
    """Top-level import names, ignoring anything nested inside a function.

    A lazy import inside a function is the whole mechanism being protected here,
    so walking the tree indiscriminately would flag exactly the correct code.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_judges_are_not_in_the_integration_extra():
    extras = _extras()
    integration = {_dist_name(r) for r in extras["integration"]}
    leaked = {n for n in integration if n.split("-")[0] in JUDGE_PACKAGES}
    assert not leaked, (
        f"{sorted(leaked)} are back in the `integration` extra. They belong in "
        "`judges`: folding them back makes every first install pay twenty "
        "minutes of dependency resolution for something most people never run."
    )


def test_the_judges_extra_exists_and_holds_all_three():
    extras = _extras()
    assert "judges" in extras, "the `judges` extra is gone"
    judges = {_dist_name(r) for r in extras["judges"]}
    for expected in ("deepeval", "ragas"):
        assert expected in judges, f"{expected} is missing from the `judges` extra"
    assert any(n.startswith("trulens") for n in judges), "trulens is missing from `judges`"


@pytest.mark.parametrize("rel", sorted(JUDGE_RUNNERS))
def test_a_judge_runner_imports_its_framework_lazily(rel: str):
    """A module-level import here would break `import eval.ragas_runner` for
    anybody who installed without the extra, and the gateway imports these
    modules to serve its diagnostics endpoints."""
    path = REPO_ROOT / rel
    if not path.exists():
        pytest.skip(f"{rel} does not exist")
    leaked = _module_level_imports(path) & JUDGE_PACKAGES
    assert not leaked, (
        f"{rel} imports {sorted(leaked)} at module level. Move it inside the "
        "function that uses it, behind a try/ImportError that names the extra: "
        "without that, a platform installed without `judges` cannot import this "
        "module at all."
    )


def test_no_other_module_reaches_for_a_judge():
    """Outside the runners, nothing may touch a judge even lazily: a judge
    reached from a request path would turn a missing optional extra into a
    surprise 500 somewhere unrelated."""
    offenders: list[str] = []
    for directory in ("core", "adapters", "services"):
        for path in sorted((REPO_ROOT / directory).rglob("*.py")):
            rel = str(path.relative_to(REPO_ROOT))
            text = path.read_text(encoding="utf-8")
            for pkg in JUDGE_PACKAGES:
                # `from eval.trulens_runner import …` is the sanctioned route and
                # names the runner rather than the framework.
                if f"import {pkg}" in text or f"from {pkg}" in text:
                    offenders.append(f"{rel} -> {pkg}")
    assert not offenders, (
        "these reach a judge framework directly instead of going through a "
        f"runner in eval/: {offenders}"
    )
