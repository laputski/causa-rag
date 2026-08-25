"""services/ingestion/cli.py resolves the structure
parser from the registry by "<pack_id>/<parser_id>", with no static
domain_packs import anywhere in the file (defense in depth, independent of
the broader domain-neutrality scan in test_p1_guardian.py).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.domain.loader import load_pack
from core.registry import registry
from services.ingestion.cli import _read_file

_CLI_FILE = Path(__file__).parents[2] / "services" / "ingestion" / "cli.py"
# A committed 4 KB fixture, copied from the reference corpus.
# It used to be read from a gitignored corpus tree behind a
# skipif, which meant these tests silently stopped running on any machine
# without that tree. Losing structure-parser coverage without noticing is a
# worse outcome than carrying 4 KB in the repository.
_CORPUS_FILE = Path(__file__).parents[1] / "fixtures" / "corpus" / "article_44.txt"
_MANUAL_FILE = Path(__file__).parents[1] / "fixtures" / "corpus" / "section_4_2_1.txt"


def test_cli_source_has_no_static_domain_packs_import() -> None:
    tree = ast.parse(_CLI_FILE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("domain_packs"):
            pytest.fail(f"static 'from {node.module} import ...' found in cli.py")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("domain_packs"):
                    pytest.fail(f"static 'import {alias.name}' found in cli.py")


def test_read_file_without_parser_leaves_structure_none() -> None:
    doc = _read_file(_CORPUS_FILE, structure_parser_fn=None)
    assert doc.structure is None


def test_read_file_resolves_real_parser_via_registry() -> None:
    # The subject used to be a pack carrying one installation's own subject
    # area, which has moved to that installation's tree. The mechanism checked is the same: the CLI
    # resolves a parser from the registry by id and knows nothing about which
    # pack put it there.
    load_pack("manuals", registry, {})
    parser_fn = registry.resolve("structure_parser", "manual_section")

    doc = _read_file(_MANUAL_FILE, structure_parser_fn=parser_fn)

    assert doc.structure is not None
    assert doc.structure.children[0].node_id == "4.2.1"
    assert doc.structure.children[0].title == "Замена детектора"
