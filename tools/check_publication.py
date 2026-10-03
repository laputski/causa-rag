"""Keep names that must not be published out of the tree and its history.

This repository is public, and the installations it is developed against hold
other people's realms. A realm's name, the name of one of its datasets, or the
size of its corpus says who the platform is being used for, and one of each
reached a pushed commit message before anything checked.

The names are a list this repository does not contain, because a list of what
must not be published is itself the thing that must not be published. It is
read from the environment (`PUBLICATION_DENYLIST`, one regular expression per
line) or from `.publication-denylist` at the root, which git ignores.

Three places are read: every tracked file as it stands, every line a commit
in the range adds, and every commit message in the range. The second matters
because a name added in one commit and removed in the next is published all
the same.

    python3 -m tools.check_publication                 # tree, and origin/main..HEAD
    python3 -m tools.check_publication --commits A..B
    python3 -m tools.check_publication --show          # print what matched, locally only
    python3 -m tools.check_publication --install-hook  # run before every push

What matched is never printed unless asked for: this runs in CI, whose log is
as public as the tree. A finding names the place and the number of the
expression. Without a list nothing is checked and the output says so; with
`--require-list` that is a failure, which is how the hook runs it.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIST_FILE = ".publication-denylist"
ENV = "PUBLICATION_DENYLIST"

NO_LIST = 2
FOUND = 1


@dataclass(frozen=True)
class Finding:
    where: str
    expression: int
    matched: str

    def render(self, show: bool) -> str:
        said = f": {self.matched!r}" if show else ""
        return f"{self.where}  (expression {self.expression}){said}"


def expressions(text: str) -> list[re.Pattern[str]]:
    """One expression per line; blank lines and lines starting with # are not."""
    return [re.compile(line.strip(), re.IGNORECASE)
            for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def load_expressions() -> list[re.Pattern[str]]:
    stated = os.getenv(ENV, "")
    path = ROOT / LIST_FILE
    if not stated.strip() and path.is_file():
        stated = path.read_text(encoding="utf-8")
    return expressions(stated)


def scan(named_lines: Iterable[tuple[str, str]], patterns: list[re.Pattern[str]]) -> list[Finding]:
    """Every match in the given (place, line) pairs."""
    found = []
    for where, line in named_lines:
        for number, pattern in enumerate(patterns, start=1):
            hit = pattern.search(line)
            if hit:
                found.append(Finding(where, number, hit.group(0)))
    return found


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          errors="replace", check=True).stdout


def tree_lines() -> Iterable[tuple[str, str]]:
    for name in _git("ls-files", "-z").split("\0"):
        path = ROOT / name
        if not name or not path.is_file():
            continue
        yield name, name  # a file's own name is published too
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # not text
        for number, line in enumerate(text.splitlines(), start=1):
            yield f"{name}:{number}", line


def commit_lines(commits: str) -> Iterable[tuple[str, str]]:
    """Each commit's message, and each line it adds."""
    for sha in _git("rev-list", commits).split():
        short = sha[:7]
        for line in _git("show", "-s", "--format=%B", sha).splitlines():
            yield f"commit {short} message", line
        for line in _git("show", "--format=", "--unified=0", sha).splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                yield f"commit {short} added line", line[1:]


def default_range() -> str | None:
    try:
        _git("rev-parse", "--verify", "--quiet", "origin/main")
    except subprocess.CalledProcessError:
        return None
    return "origin/main..HEAD"


_HOOK = """#!/bin/sh
# Installed by `python3 -m tools.check_publication --install-hook`.
exec python3 -m tools.check_publication --require-list
"""


def install_hook() -> Path:
    hook = ROOT / ".git" / "hooks" / "pre-push"
    hook.write_text(_HOOK, encoding="utf-8")
    hook.chmod(0o755)
    return hook


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--commits", help="a revision range; default origin/main..HEAD")
    parser.add_argument("--show", action="store_true", help="print what matched; never in CI")
    parser.add_argument("--require-list", action="store_true",
                        help="fail when there is no list to check against")
    parser.add_argument("--install-hook", action="store_true", help="check before every push")
    args = parser.parse_args(argv)

    if args.install_hook:
        print(f"installed {install_hook().relative_to(ROOT)}")
        return 0

    patterns = load_expressions()
    if not patterns:
        print(f"publication check NOT RUN: no list in ${ENV} or {LIST_FILE}")
        return NO_LIST if args.require_list else 0

    commits = args.commits or default_range()
    found = scan(tree_lines(), patterns)
    if commits:
        found += scan(commit_lines(commits), patterns)
    if not found:
        scope = f"the tree and {commits}" if commits else "the tree"
        print(f"publication check: nothing from the list in {scope} ({len(patterns)} expressions)")
        return 0
    for finding in found:
        print(finding.render(args.show))
    print(f"\n{len(found)} place(s) carry something from the list. "
          "Run with --show locally to see what; a pushed commit needs its history rewritten.")
    return FOUND


if __name__ == "__main__":
    sys.exit(main())
