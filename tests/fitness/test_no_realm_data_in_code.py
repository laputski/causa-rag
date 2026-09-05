"""A Realm's data lives in the database, and its names stay out of the code.

This repository is developed in public while the same code runs privately
against real Realms with real corpora. Those two facts collide in exactly one
place: a Realm's identifiers leaking into a fixture, a default, a hostname or a
comment, and travelling to the public tree with the next export.

`tools/check_publishable.py` guards against that by naming the identifiers it
knows. That is the right tool for material already known to be sensitive, and it
is structurally unable to catch the next Realm: a name nobody has written a
pattern for yet passes it. Every leak found so far arrived in a shape the
patterns did not anticipate. A Realm's name reached the tree as a compose
service name, as a test hostname, as a custom model name in a proxy config, as a
corpus id, as a dataset filename and as a prompt id, and the patterns of the day
described none of those positions.

So this works the other way around: the set of Realm-shaped names allowed in
code is closed, and anything outside it fails. Adding a Realm to the platform
needs no change here, because a real Realm is created through the API and lives
in Mongo. Adding one to the *code* does, which is the point.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Directories holding nothing this rule governs: dependencies, caches, build
# output, one installation's own packs, and the private design notes and
# research that never publish.
SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "dist", "build", "test-reports", ".deepeval",
    ".claude", ".specify", "htmlcov", "packs.local", "corpus",
    "lat.md", "research", "specs", "governance", "architecture", "infra",
}
SKIP_PREFIXES = ("eval/results", "eval/judgments", "eval/golden", "docs/ru")

# Anything the export already withholds is out of scope: those files stay in the
# private tree by design, and the per-installation prompt files in particular
# hold the Realm ids of the installations they belong to, which is where those
# ids belong.
#
# The exclusion list is read from the export itself and never copied, so the
# two cannot drift. On the public repository `tools/publish_export.py` does not
# exist, and there nothing is excluded, because everything present has already
# been published.
try:
    from tools.publish_export import is_excluded as _export_excludes
except ImportError:  # the public tree
    def _export_excludes(rel: str) -> bool:  # noqa: D103
        return False

SUFFIXES = {".py", ".ts", ".tsx", ".js", ".json", ".yaml", ".yml"}

# The file whose whole purpose is to name the forbidden strings, and the tests
# that assert those names are absent. Each would report itself forever.
SELF_EXEMPT = {
    "tools/check_publishable.py",
    "tests/unit/test_demo_corpus.py",
    "tests/fitness/test_no_realm_data_in_code.py",
}

# Every Realm id a fixture, example or default may use.
#
# `demo` is the realm the platform ships. `acme` is the stand-in for "some other
# realm", which is what most scoping tests need. The rest are shapes a test
# asserts on directly: a generated e2e id, a rename collision, a placeholder in
# the connector guide.
ALLOWED_REALM_IDS = {
    # "proving-ground" joins "demo" for the same reason: both name a realm this
    # repository invents and ships a seed for, so both are written in code by
    # design. Neither came off anybody's installation.
    "demo", "demo-2", "proving-ground", "acme", "acme-2",
    "r", "other", "other-realm", "my-realm", "quickstart",
    "e2e-x", "realm_a", "realm_b", "", "null", "undefined",
}

# Every corpus id a fixture, example or default may use. A corpus id names a
# body of documents, so a real one names what an installation holds.
ALLOWED_CORPUS_IDS = {
    "handbook", "handbook_01", "demo_corpus", "demo", "default", "manuals",
    "acme-corpus", "acme_v2", "external_docs", "graphrag_docs", "docs",
    "c", "active", "gone", "orphan", "missing", "empty_corpus", "other",
    "brand_new_corpus", "test_corpus", "corpus", "",
    # The two corpora this repository writes and ships for the proving ground.
    # They were invisible to this guard until the seed named its fields: the
    # pattern below wants the word `corpus_id` beside the value, and the seed
    # listed them as bare tuple positions, so a corpus called anything at all
    # passed. Found by planting one.
    "base-ru", "base-en",
}

# A dotted lowerCamel path is an i18n key and not an id: `corpusId:
# 'runPage.config.corpusIdHint'` names a translation, not a corpus.
_I18N_KEY_RE = re.compile(r"[a-z][A-Za-z0-9]*(?:\.[a-zA-Z][A-Za-z0-9]*){2,}")

_FIELD_RE = re.compile(
    r"""\b(realm_?id|corpus_?id)["']?\s*[:=]\s*["']([A-Za-z0-9_.\-]*)["']""",
    re.IGNORECASE,
)


def _git_ignored() -> set[str]:
    """Paths git will never commit, and which therefore cannot leak.

    Running the gateway inside a checkout writes this installation's prompts to
    `prompts/*.json`, which `.gitignore` covers precisely so they stay local.
    Scanning the filesystem without asking git turned that into a failure over
    files no push could ever carry.

    Degrades to "nothing is ignored" outside a git working tree, which is the
    stricter reading and the right one when the question cannot be answered."""
    import subprocess
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "--others", "--ignored",
             "--exclude-standard", "-z"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    return {line for line in out.stdout.split("\0") if line}


def _files() -> list[Path]:
    ignored = _git_ignored()
    out: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in SUFFIXES:
            continue
        rel = path.relative_to(REPO_ROOT)
        if set(rel.parts) & SKIP_DIRS:
            continue
        rel_str = str(rel)
        if rel_str in SELF_EXEMPT or rel_str.startswith(SKIP_PREFIXES):
            continue
        if _export_excludes(rel_str) or rel_str in ignored:
            continue
        out.append(path)
    return out


def _violations(field_prefix: str, allowed: set[str]) -> list[str]:
    found: list[str] = []
    for path in _files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = path.relative_to(REPO_ROOT)
        for line_no, line in enumerate(text.splitlines(), start=1):
            for match in _FIELD_RE.finditer(line):
                field = match.group(1).lower().replace("_", "")
                if not field.startswith(field_prefix):
                    continue
                value = match.group(2)
                if value.lower() in allowed or _I18N_KEY_RE.fullmatch(value):
                    continue
                found.append(f"{rel}:{line_no}: {field} = {value!r}")
    return found


# @lat: [[publication#Preparing the repository for public release#The publication guard runs before any copy, not after#The guard can only catch names somebody wrote a pattern for]]
def test_no_unlisted_realm_id_appears_in_code() -> None:
    """A Realm id in code is a Realm that somebody hardcoded.

    Real Realms are created through the API and stored in Mongo, so the only
    ones that belong in a source file are the invented ones a fixture needs.
    Widen `ALLOWED_REALM_IDS` deliberately when a fixture needs a new invented
    name; never to accommodate a name that came off a running installation."""
    bad = _violations("realmid", ALLOWED_REALM_IDS)
    assert not bad, "Realm ids not on the allow-list:\n  " + "\n  ".join(bad)


# @lat: [[publication#Preparing the repository for public release#The publication guard runs before any copy, not after#The guard can only catch names somebody wrote a pattern for]]
def test_no_unlisted_corpus_id_appears_in_code() -> None:
    """A corpus id names a body of documents somebody actually holds.

    Two corpus ids naming an installation's subject area sat in this tree as
    fixtures in the client SDK's own tests, and said what that installation
    holds as plainly as the documents would have."""
    bad = _violations("corpusid", ALLOWED_CORPUS_IDS)
    assert not bad, "corpus ids not on the allow-list:\n  " + "\n  ".join(bad)
