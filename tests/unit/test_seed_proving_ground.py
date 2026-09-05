"""The proving ground is its own realm, and it says what it is.

The demo realm answers "does the installation work", and it can only answer that
while it stays healthy. A realm holding data broken on purpose has to be a
different one, and it has to carry a mark: without one its diagnostics read as a
broken installation, and a green proving ground would mean the guards had
stopped firing on the defects put there for them.

Seeded through the same import path a person uses by hand, so the seed keeps
exercising that code instead of writing documents behind its back.
"""
from __future__ import annotations

import asyncio
import pathlib
from typing import Any
from unittest.mock import patch

import pytest

from tools import seed_proving_ground as seed


class _EmptyMongo:
    """A database holding nothing: what a proving ground is seeded onto."""

    def __init__(self) -> None:
        self.written: dict[str, list[dict[str, Any]]] = {}

    async def find_one(self, collection: str, query: dict | None = None) -> None:
        return None

    async def find_many(self, collection: str, query: dict | None = None, **kw: Any) -> list:
        return []

    async def insert_one(self, collection: str, doc: dict) -> None:
        self.written.setdefault(collection, []).append(doc)

    async def update_one(self, collection: str, query: dict, update: dict) -> None:
        pass

    async def upsert_one(self, collection: str, query: dict, doc: dict) -> None:
        self.written.setdefault(collection, []).append(doc)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> _EmptyMongo:
    import adapters.mongodb as mdb

    empty = _EmptyMongo()
    for name in ("find_one", "find_many", "insert_one", "update_one", "upsert_one"):
        monkeypatch.setattr(mdb, name, getattr(empty, name))
    return empty


def test_the_bundle_travels_the_same_route_a_person_would_use(store: _EmptyMongo) -> None:
    """Through `import_realm`, so the seed cannot drift from the import path.

    The demo seed is built this way for the same reason: an example loaded by a
    private route stops being evidence that the public one works.
    """
    with patch("core.prompt_store.prompt_store.sync_from"):
        asyncio.run(seed.seed(do_ingest=False))
    realms = store.written.get("realms", [])
    assert realms, "no realm was written"
    assert realms[0]["id"] == "proving-ground"


def test_the_realm_says_it_is_broken_on_purpose(store: _EmptyMongo) -> None:
    """Without the mark, red diagnostics there read as a broken installation,
    and the installation check would report on a realm built to fail."""
    with patch("core.prompt_store.prompt_store.sync_from"):
        asyncio.run(seed.seed(do_ingest=False))
    realm = store.written["realms"][0]
    assert realm["purpose"] == "proving_ground"
    assert realm["description"], "the mark alone explains nothing to a reader"


def test_the_installation_check_passes_this_realm_over() -> None:
    """The peer here is named so that it sorts *after* the proving ground.

    Written first against a peer called "demo", which sorts before it, so the
    check picked the right realm by alphabet and the assertion held whether or
    not the mark was there at all. A test that passes for a reason other than
    the one it names is a test that will keep passing once the reason breaks.
    """
    from tools.doctor import _subject_realm

    realm = seed.build_bundle()["realm"]
    assert realm["id"] < "workshop", "the peer no longer sorts after the realm under test"
    assert _subject_realm([realm, {"id": "workshop"}]) == "workshop"


def test_both_corpora_are_seeded_and_neither_is_broken() -> None:
    """The two halves every mutation is measured as a difference from. A seed
    that laid a damaged corpus would leave nothing to compare against."""
    bundle = seed.build_bundle()
    ids = [c["corpus_id"] for c in bundle["corpora"]]
    assert ids == ["base-ru", "base-en"]


def test_each_corpus_names_the_analyser_its_index_is_built_with() -> None:
    """An index carries its analyser for life and every query inherits it, so a
    corpus loaded under the wrong one is a defect no later setting undoes.
    Naming it makes a deliberate mismatch a visible act and not a forgotten flag.
    """
    languages = {corpus_id: language for corpus_id, _, language, _ in seed.CORPORA}
    assert languages == {"base-ru": "ru_be", "base-en": "en"}


def test_every_question_set_travels_with_the_realm() -> None:
    """A corpus arriving without the set that queries it measures nothing: every
    retrieval metric comes back zero, and zero is what a broken system and a
    broken pairing both look like."""
    bundle = seed.build_bundle()
    corpora = {c["corpus_id"] for c in bundle["corpora"]}
    datasets = {d["filename"].split(".")[0] for d in bundle["datasets"]}
    assert corpora == datasets
    for dataset in bundle["datasets"]:
        assert dataset["questions"], f"{dataset['filename']} carries no questions"


def test_the_prompt_asks_for_the_marker_the_platform_substitutes() -> None:
    """A prompt asking for a marker the substitution does not recognise leaves
    the answer carrying a label that means nothing beside the context list, and
    `computed_citations` empty. The demo realm shipped that way once, asking for
    "[section N]" while the code rewrites "Fragment N" and nothing else.

    The template carries a literal "N" where the model writes a number, so the
    check puts a number there first and then asserts the rewriting happens.
    """
    from core.citation import substitute_fragment_markers
    from core.models import SourceRef

    refs = [
        SourceRef(doc_id="proving-ground/base-ru/01", chunk_id="c1", score=0.9,
                  structural_path="document/section[1 Назначение]"),
        SourceRef(doc_id="proving-ground/base-ru/02", chunk_id="c2", score=0.8,
                  structural_path="document/section[2 Порядок]"),
    ]
    template = seed.build_bundle()["prompts"][0]["template"].replace("N)", "1)")
    assert substitute_fragment_markers(template, refs) != template, (
        "the prompt asks for a citation marker core/citation.py does not rewrite"
    )


def test_seeding_twice_changes_nothing_the_second_time(store: _EmptyMongo) -> None:
    """Idempotent, so a person can run it without wondering."""
    import adapters.mongodb as mdb

    with patch("core.prompt_store.prompt_store.sync_from"):
        asyncio.run(seed.seed(do_ingest=False))
    first = len(store.written.get("realms", []))

    async def find_existing(collection: str, query: dict | None = None) -> dict | None:
        return {"id": "proving-ground"} if collection == "realms" else None

    with patch.object(mdb, "find_one", find_existing), patch("core.prompt_store.prompt_store.sync_from"):
        asyncio.run(seed.seed(do_ingest=False))
    assert len(store.written.get("realms", [])) == first, "the realm was written twice"


def test_it_never_touches_the_demo_realm(store: _EmptyMongo) -> None:
    """The demo has its own work, and damaged data would destroy exactly the
    thing it exists to show."""
    with patch("core.prompt_store.prompt_store.sync_from"):
        asyncio.run(seed.seed(do_ingest=False))
    for collection, docs in store.written.items():
        for doc in docs:
            assert doc.get("realm_id") != "demo", f"{collection}: wrote into the demo realm"
            assert doc.get("id") != "demo", f"{collection}: wrote the demo realm itself"


def test_force_removes_this_realm_and_leaves_every_other_one_standing() -> None:
    """`--force` deletes, so where it stops has to be observed and not read.

    Two realms go in, both carrying documents in every collection the reseed
    touches; only the proving ground's are allowed to leave.
    """
    import adapters.mongodb as mdb

    collections = ["realms", "prompts", "generation_presets", "datasets", "external_rags", "corpora"]
    docs: dict[str, list[dict[str, Any]]] = {}
    for name in collections:
        key = "id" if name == "realms" else "realm_id"
        docs[name] = [
            {"_id": f"{name}-pg", key: "proving-ground"},
            {"_id": f"{name}-demo", key: "demo"},
        ]

    async def find_many(collection: str, query: dict | None = None, **kw: Any) -> list:
        return [d for d in docs.get(collection, [])
                if all(d.get(k) == v for k, v in (query or {}).items())]

    async def delete_one(collection: str, query: dict) -> None:
        docs[collection] = [d for d in docs[collection] if d["_id"] != query["_id"]]

    with patch.object(mdb, "find_many", find_many), patch.object(mdb, "delete_one", delete_one):
        removed = asyncio.run(seed._drop_realm(seed.REALM_ID))

    assert removed == len(collections), f"reported {removed} removals of {len(collections)}"
    survivors = sorted(d["_id"] for group in docs.values() for d in group)
    assert survivors == sorted(f"{name}-demo" for name in collections), (
        f"documents of another realm were removed: {survivors}"
    )


def test_force_says_how_many_documents_it_removed() -> None:
    """A tolerant `except` around the deletion would otherwise let a reseed
    report success over documents that are all still there."""
    import adapters.mongodb as mdb

    async def refusing_find_many(collection: str, query: dict | None = None, **kw: Any) -> list:
        raise RuntimeError("collection unreachable")

    with patch.object(mdb, "find_many", refusing_find_many):
        assert asyncio.run(seed._drop_realm(seed.REALM_ID)) == 0


def test_the_corpora_the_seed_names_are_actually_on_disk() -> None:
    """The seed's list and the repository's contents are two things, and a
    mismatch surfaces only when somebody runs the ingest, which needs a whole
    stack up. Cheap to check here; expensive to discover there."""
    for corpus in seed.CORPORA:
        directory = seed.GROUND_DIR / corpus.directory
        assert directory.is_dir(), f"{corpus.corpus_id}: {directory} does not exist"
        assert sorted(directory.glob("*.md")), f"{corpus.corpus_id}: no documents in {directory}"
        golden = seed.GOLDEN_DIR / f"{corpus.corpus_id}.v1.fast.jsonl"
        assert golden.is_file(), f"{corpus.corpus_id}: no question set at {golden}"


def test_a_missing_corpus_stops_the_seed_instead_of_being_skipped(
    store: _EmptyMongo, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A silent skip leaves a realm holding a question set and no documents.

    Every retrieval metric then reads zero, which is what a broken system and an
    empty corpus both look like, and the reseed that caused it reported success.
    """
    monkeypatch.setattr(seed, "GROUND_DIR", tmp_path / "nothing-here")
    with patch("core.prompt_store.prompt_store.sync_from"), pytest.raises(SystemExit) as raised:
        asyncio.run(seed.seed(do_ingest=True))
    assert "nothing-here" in str(raised.value)


def test_every_string_a_reader_sees_is_in_one_language() -> None:
    """A realm's name and description are stored strings, shown verbatim.

    Nothing translates them: the interface reads them out of the realm document
    in whatever locale it is running, exactly as it does for the demo realm,
    whose name is "Demo" and whose description is English. The proving ground
    shipped named "Полигон", so an English reader met a Russian name on the
    realm switcher and the overview.

    The corpora are Russian and English on purpose and their *documents* stay
    so; this is about the labels around them. The same rule the interface's own
    coverage test applies to its English locale file.
    """
    bundle = seed.build_bundle()
    labels = [bundle["realm"]["name"], bundle["realm"]["description"]]
    labels += [corpus["description"] for corpus in bundle["corpora"]]
    labels += [bundle["prompts"][0]["name"], bundle["prompts"][0]["description"]]
    cyrillic = [text for text in labels if any("Ѐ" <= ch <= "ӿ" for ch in text)]
    assert cyrillic == [], f"labels a reader sees in any locale, written in one of them: {cyrillic}"
