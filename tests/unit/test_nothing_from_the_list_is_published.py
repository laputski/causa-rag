"""A name on the list is found in the tree, in what a commit adds, and in its message.

The list is whatever the installation says must not be published; the name
used here is invented. Found the hard way: a realm's name, the name of its
dataset and the size of its corpus went out in a commit message, and the
history had to be rewritten.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools import check_publication as P

LISTED = "acme-realm"


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t", "HOME": str(root), "PATH": "/usr/bin:/bin:/usr/local/bin"})


@pytest.fixture
def repo(tmp_path: Path, monkeypatch) -> Path:
    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "clean.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text(f"{P.LIST_FILE}\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "A clean start")
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", "HEAD")
    monkeypatch.setattr(P, "ROOT", tmp_path)
    monkeypatch.delenv(P.ENV, raising=False)
    return tmp_path


def _commit(repo: Path, name: str, text: str, message: str) -> None:
    (repo / name).write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)


def _run(repo: Path, capsys, *argv: str) -> tuple[int, str]:
    code = P.main(list(argv))
    return code, capsys.readouterr().out


def _listed(repo: Path) -> None:
    (repo / P.LIST_FILE).write_text(f"# what must not go out\n{LISTED}\n", encoding="utf-8")


def test_a_clean_tree_and_history_pass(repo, capsys) -> None:
    _listed(repo)
    _commit(repo, "more.py", "y = 2\n", "Another clean commit")
    code, out = _run(repo, capsys)
    assert code == 0 and "nothing from the list" in out


def test_a_name_in_a_tracked_file_is_found(repo, capsys) -> None:
    _listed(repo)
    _commit(repo, "notes.md", f"measured on {LISTED} yesterday\n", "Notes")
    code, out = _run(repo, capsys)
    assert code == P.FOUND and "notes.md:1" in out


def test_a_name_in_a_commit_message_is_found(repo, capsys) -> None:
    _listed(repo)
    _commit(repo, "more.py", "y = 2\n", f"Checked that {LISTED} still works")
    code, out = _run(repo, capsys)
    assert code == P.FOUND and "message" in out


def test_a_name_added_and_then_removed_is_found(repo, capsys) -> None:
    """Published all the same: the commit that added it is in the history."""
    _listed(repo)
    _commit(repo, "notes.md", f"{LISTED}\n", "Notes")
    _commit(repo, "notes.md", "nothing here\n", "Notes, tidied")
    code, out = _run(repo, capsys)
    assert code == P.FOUND and "added line" in out and "notes.md:" not in out


def test_a_name_in_a_file_s_own_name_is_found(repo, capsys) -> None:
    _listed(repo)
    _commit(repo, f"{LISTED}.yml", "a: 1\n", "A config")
    assert _run(repo, capsys)[0] == P.FOUND


def test_what_matched_is_not_printed_unless_asked_for(repo, capsys) -> None:
    """This runs in CI, whose log is as public as the tree."""
    _listed(repo)
    _commit(repo, "notes.md", f"measured on {LISTED.upper()} yesterday\n", "Notes")
    _, out = _run(repo, capsys)
    assert LISTED not in out.lower()
    _, shown = _run(repo, capsys, "--show")
    assert LISTED in shown.lower()


def test_without_a_list_nothing_is_checked_and_the_output_says_so(repo, capsys) -> None:
    _commit(repo, "notes.md", f"{LISTED}\n", "Notes")
    code, out = _run(repo, capsys)
    assert code == 0 and "NOT RUN" in out


def test_the_hook_refuses_to_pass_without_a_list(repo, capsys) -> None:
    """A push that was checked against nothing was not checked."""
    assert _run(repo, capsys, "--require-list")[0] == P.NO_LIST


def test_the_list_is_read_from_the_environment_first(repo, capsys, monkeypatch) -> None:
    monkeypatch.setenv(P.ENV, LISTED)
    _commit(repo, "notes.md", f"{LISTED}\n", "Notes")
    assert _run(repo, capsys)[0] == P.FOUND


def test_the_hook_is_installed_and_runs_the_check(repo) -> None:
    hook = P.install_hook()
    assert hook.name == "pre-push" and hook.stat().st_mode & 0o111
    assert "--require-list" in hook.read_text(encoding="utf-8")
