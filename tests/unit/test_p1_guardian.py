"""Domain-neutrality guardian: core/ must contain NO domain-specific terms.

Legal, medical, and other domain terms are only allowed inside domain_packs/.
This test blocks CI if any domain term leaks into the core layer.
"""
from __future__ import annotations

import re
from pathlib import Path

_CORE_DIR = Path(__file__).parents[2] / "core"
_SERVICES_DIR = Path(__file__).parents[2] / "services"

# A direct static import of a concrete domain_packs.* module from services/ is
# exactly the violation that let a hardcoded
# `from domain_packs.<pack>.structure_parser import ...` slip into
# services/ingestion/cli.py undetected
# for a full stage — this is narrower than _LEGAL_TERMS
# above (it's not about vocabulary, only about a static cross-package import;
# services/ legitimately handles domain-flavored strings in API payloads).
_DOMAIN_IMPORT = re.compile(r"^\s*(from|import)\s+domain_packs(\.|\s|$)")

# Terms that belong ONLY in domain_packs — never in core/
#
# They stay Russian: a domain leak into the platform core would arrive in the
# vocabulary of the corpora it was built against, so those are the strings worth
# looking for. Translating them would leave the guard matching nothing.
_LEGAL_TERMS = re.compile(
    r"\b("
    r"НПА|нормативн|правовой акт|юридич|правовая сила"
    r"|постановлени[еяю]|закон\b|кодекс|декрет|указ\b"
    r"|регламент|анализ\b.*(?:правов|юрид)"
    r"|legal|statute|regulation(?:s)?\b|ordinance"
    r"|закрытый вопрос|открытый вопрос|навигационный вопрос"
    r")",
    re.IGNORECASE | re.UNICODE,
)


def _python_files(directory: Path) -> list[Path]:
    return sorted(directory.rglob("*.py"))


def test_no_legal_terms_in_core() -> None:
    violations: list[str] = []
    for pyfile in _python_files(_CORE_DIR):
        lines = pyfile.read_text(encoding="utf-8").splitlines()
        for lineno, line in enumerate(lines, start=1):
            # skip comments and docstrings that may reference domain for educational purposes
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            if _LEGAL_TERMS.search(line):
                violations.append(f"{pyfile.relative_to(_CORE_DIR.parent)}:{lineno}: {line.strip()}")

    if violations:
        formatted = "\n  ".join(violations)
        raise AssertionError(
            f"Domain neutrality violated — domain terms found in core/:\n  {formatted}\n\n"
            "Move them to domain_packs/."
        )


def test_no_direct_domain_pack_imports_in_services() -> None:
    violations: list[str] = []
    for pyfile in _python_files(_SERVICES_DIR):
        for lineno, line in enumerate(pyfile.read_text(encoding="utf-8").splitlines(), start=1):
            if _DOMAIN_IMPORT.match(line):
                violations.append(f"{pyfile.relative_to(_SERVICES_DIR.parent)}:{lineno}: {line.strip()}")

    if violations:
        formatted = "\n  ".join(violations)
        raise AssertionError(
            f"Domain neutrality violated — direct domain_packs import in services/:\n  {formatted}\n\n"
            "Resolve domain components via core.domain.loader + the registry instead."
        )


# A filesystem path to a concrete corpus, hardcoded in platform code, is the
# defect that was removed. One
# Realm's corpus directory was baked into services/api_gateway/routers/
# experiments.py and consulted for EVERY Realm's run, against the platform's
# own disk rather than the index the served system searches. Because the
# directory is gitignored, a fresh checkout silently classified every question
# "uncovered" and zeroed retrieval metrics with no error raised anywhere.
# the design notes records four earlier incidents from the same
# coupling; this guard makes a fifth fail the build instead of shipping.
_CORPUS_PATH_LITERAL = re.compile(r"""["']corpus["']\s*/|["'][^"']*corpus/[a-z_]+["']""")


def test_no_hardcoded_corpus_paths_in_core_or_services() -> None:
    violations: list[str] = []
    for directory in (_CORE_DIR, _SERVICES_DIR):
        for pyfile in _python_files(directory):
            for lineno, line in enumerate(pyfile.read_text(encoding="utf-8").splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if _CORPUS_PATH_LITERAL.search(line):
                    violations.append(
                        f"{pyfile.relative_to(directory.parent)}:{lineno}: {stripped}"
                    )

    if violations:
        formatted = "\n  ".join(violations)
        raise AssertionError(
            f"VIOLATION — hardcoded corpus path in core/ or services/:\n  {formatted}\n\n"
            "A corpus belongs to one Realm and lives wherever that Realm's resources say.\n"
            "Resolve coverage against the INDEX via core/eval/ref_resolution.py instead."
        )


# ── "stand outside the hot path" ─────────────────────────────
# the design notes states the invariant: no path serving a real
# user's query may depend on the platform's database being reachable. It was
# violated once already — the chat path fetched retrieval pins from MongoDB
# on every message, and a failed lookup silently changed the answer with
# nothing reporting it. These two guards make a repeat fail the build.

_MONGO_IMPORT = re.compile(r"^\s*(from|import)\s+adapters\.mongodb(\.|\s|$)|from\s+adapters\s+import\s+.*mongodb")


def test_no_mongo_access_from_core() -> None:
    """core/ is the layer a served system is grown from. If a store lookup
    reaches it, every system built from this template inherits a dependency
    on the platform being up at query time."""
    offenders = [
        f"{path.relative_to(_CORE_DIR)}:{i}"
        for path in _python_files(_CORE_DIR)
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if _MONGO_IMPORT.search(line)
    ]
    assert not offenders, (
        "core/ must not reach the platform's database. Resolve the data in "
        "services/ and pass it in at construction, the way qdrant_cfg and "
        "ref_resolver already are:\n  " + "\n  ".join(offenders)
    )


def test_chat_path_applies_no_retrieval_pins() -> None:
    """Chat is the closest thing the platform has to a real request path, so
    it is the one place the invariant is easiest to break and costliest to
    break. Pins belong to an opt-in experiment run, never to a live query."""
    main_py = Path(__file__).parents[2] / "services" / "api_gateway" / "main.py"
    offenders = [
        f"main.py:{i}: {line.strip()}"
        for i, line in enumerate(main_py.read_text(encoding="utf-8").splitlines(), 1)
        if "_load_active_pins" in line or "retrieval_pins=" in line
    ]
    assert not offenders, (
        "the chat path must not load or pass retrieval pins:\n  "
        + "\n  ".join(offenders)
    )
