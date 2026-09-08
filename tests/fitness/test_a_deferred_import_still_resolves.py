"""An import written inside a function body still names something that exists.

This platform defers imports on purpose: it is what keeps `core` importable
without the adapters installed, and what keeps a heavy optional dependency out
of the start-up path. The cost is that nothing checks them. A module-level
import that names a function nobody wrote any more fails the moment the module
is imported, so the test suite finds it at collection; one inside a function
body fails only when that function runs.

Found by making the mistake. Five rebuilding functions in the experiment
builder were replaced by protocols on the components themselves, and three
callers named them from inside function bodies: the reference server, the
faulty one, and two gateway routers. Every unit and contract test passed. The
proving ground found it, by running the live stack, eight minutes in, as an
internal server error on every question of every pair.

Nothing is imported here. The name is looked for in the source of the module
it was asked of, so a module needing a dependency this machine has not got is
checked like any other, and a third-party module is left alone.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Our own top-level packages. An import of anything else is somebody else's
# module, and what it exports is not this repository's to check.
OURS = {"core", "adapters", "services", "tools", "eval", "domain_packs", "clients"}

SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
             ".mypy_cache", ".ruff_cache", "dist", "build", "htmlcov", "packs.local"}


def _files() -> list[Path]:
    return [p for p in REPO_ROOT.rglob("*.py")
            if not any(part in SKIP_DIRS for part in p.relative_to(REPO_ROOT).parts)]


def _module_file(module: str) -> Path | None:
    """The file a dotted module name refers to, when this repository holds it."""
    if module.split(".")[0] not in OURS:
        return None
    as_module = REPO_ROOT / (module.replace(".", "/") + ".py")
    if as_module.is_file():
        return as_module
    as_package = REPO_ROOT / module.replace(".", "/") / "__init__.py"
    return as_package if as_package.is_file() else None


def _names_defined_in(path: Path) -> set[str]:
    """Everything a module makes available by name, read from its source.

    A name bound at the top level in any of the ways Python allows: a
    function, a class, an assignment, an import of its own. Conditional
    definitions count, since a caller cannot tell which branch ran.
    """
    found: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            found.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                found.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            for target in ast.walk(node):
                if isinstance(target, ast.Name):
                    found.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found.add(node.target.id)
    return found


def _deferred_imports() -> list[tuple[str, int, str, str]]:
    """Every `from <ours> import <name>` written inside a function body."""
    out: list[tuple[str, int, str, str]] = []
    for path in _files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for function in ast.walk(tree):
            if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for node in ast.walk(function):
                if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
                    continue
                if not _module_file(node.module):
                    continue
                for alias in node.names:
                    if alias.name != "*":
                        out.append((str(path.relative_to(REPO_ROOT)), node.lineno,
                                    node.module, alias.name))
    return out


def test_there_are_deferred_imports_to_check() -> None:
    """Without this the check below would pass on an empty list, which is the
    shape of a guard that has quietly stopped guarding."""
    assert len(_deferred_imports()) > 20


def test_every_deferred_import_names_something_that_exists() -> None:
    dangling = []
    known: dict[str, set[str]] = {}
    for where, line, module, name in _deferred_imports():
        if module not in known:
            known[module] = _names_defined_in(_module_file(module))  # type: ignore[arg-type]
        # A submodule of a package is a name too: `from core.eval import atlas`.
        if name in known[module] or _module_file(f"{module}.{name}"):
            continue
        dangling.append(f"{where}:{line} asks {module} for {name!r}, which it has not got")
    assert dangling == [], "\n  " + "\n  ".join(sorted(dangling))
