"""The client package has one name, and everything that mentions it agrees.

A Python package carries two names that people confuse: the distribution name
`pip install` takes, and the module name `import` takes. They live in different
files, nothing makes them agree, and a rename touches both plus every document
that quotes either.

This was renamed from `rag-platform-client` to `causa-rag-client` for the public
release, across the directory, the manifest, the imports, both connector guides
in two languages, and the worked example. A test is cheaper than checking twenty
files by hand, and it is what catches the half-rename where the import works and
the documented `pip install` line names something that does not exist.
"""
from __future__ import annotations

import pathlib
import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CLIENT_ROOT = REPO_ROOT / "clients" / "python"

DIST_NAME = "causa-rag-client"
MODULE_NAME = "causa_rag_client"

# Everything that quotes one of the two names. A file added here without being
# renamed is exactly the failure this catches.
MENTIONS = [
    "README.md",
    "docs/connector-guide.md",
    "docs/external-rag-contract.md",
    "docs/ru/connector-guide.md",
    "docs/ru/external-rag-contract.md",
    "examples/connector_quickstart/README.md",
    "examples/connector_quickstart/run_experiment.py",
    "examples/connector_quickstart/serve_app.py",
]

OLD_NAMES = ("rag-platform-client", "rag_platform_client")


# @lat: [[connector-client#Connector client library#One name, checked]]
def test_the_distribution_name_matches_the_module_directory():
    with (CLIENT_ROOT / "pyproject.toml").open("rb") as fh:
        declared = tomllib.load(fh)["project"]["name"]
    assert declared == DIST_NAME, f"pyproject declares {declared!r}"
    assert (CLIENT_ROOT / MODULE_NAME).is_dir(), (
        f"the module directory is not {MODULE_NAME}/. The distribution name and "
        "the import name are separate strings and nothing else makes them agree."
    )


# @lat: [[connector-client#Connector client library#One name, checked]]
def test_no_file_still_carries_the_old_name():
    offenders: list[str] = []
    for rel in MENTIONS + ["clients/python/pyproject.toml"]:
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for old in OLD_NAMES:
            if old in text:
                offenders.append(f"{rel} -> {old}")
    for path in sorted(CLIENT_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for old in OLD_NAMES:
            if old in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} -> {old}")
    assert not offenders, f"the old package name survives in: {offenders}"


# @lat: [[connector-client#Connector client library#One name, checked]]
def test_the_documented_import_is_the_one_that_works():
    """Both guides show `from causa_rag_client import …`. If the module were
    renamed again without the docs, that line would be the first thing a reader
    copies and the first thing that fails."""
    for rel in ("docs/connector-guide.md", "docs/ru/connector-guide.md"):
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        assert f"from {MODULE_NAME} import" in text, (
            f"{rel} does not show an import from {MODULE_NAME}"
        )


# @lat: [[connector-client#Connector client library#One name, checked]]
def test_the_guide_offers_an_install_that_works_before_the_first_release():
    """`pip install causa-rag-client` resolves nothing until the package is on
    PyPI. Documenting only that line would fail every reader up to the first
    release, so the guide also names the path that works from a checkout."""
    text = (REPO_ROOT / "docs" / "connector-guide.md").read_text(encoding="utf-8")
    assert "clients/python" in text, (
        "the connector guide gives no way to install the client before the "
        "first PyPI release"
    )
