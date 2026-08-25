"""The demo realm.

The file the "Open the demo" button loads on a first start. It is also the check
that realm import from JSON works at all: there is deliberately no separate
"create an example" mechanism, because that would let the example and the import
path diverge, and the example would stop being a check.
"""
import json
import pathlib

import pytest

import adapters.mongodb as mdb
from core.models import SourceRef
from services.api_gateway.routers import realms as R

DEMO = pathlib.Path(__file__).parents[2] / "ui" / "public" / "demo.realm.json"

# Two source refs, enough for a marker to have something to resolve against.
_REFS = [
    SourceRef(doc_id="demo_handbook/01", chunk_id="c1", score=0.9,
              structural_path="document/section[1. Purchasing approvals]"),
    SourceRef(doc_id="demo_handbook/02", chunk_id="c2", score=0.8,
              structural_path="document/section[2. Travel and expenses]"),
]


@pytest.fixture
def bundle() -> dict:
    return json.loads(DEMO.read_text(encoding="utf-8"))


class _EmptyMongo:
    """An empty database: the demo loads onto an installation holding nothing."""

    def __init__(self):
        self.written: dict[str, list] = {}

    async def find_one(self, collection, query=None):
        return None

    async def find_many(self, collection, query=None):
        return []

    async def insert_one(self, collection, doc):
        self.written.setdefault(collection, []).append(doc)

    async def update_one(self, collection, query, update):
        pass


# @lat: [[overview#Overview — the realm's own page, and the first screen of all#First run: three doors, not a dimmed interface]]
def test_demo_bundle_imports_cleanly(bundle, monkeypatch):
    """The demo travels the same route as any installation-to-installation
    transfer. If the format drifts from what `import_realm` accepts, the very
    first button on the very first screen hits a parse error."""
    import asyncio

    store = _EmptyMongo()
    for name in ("find_one", "find_many", "insert_one", "update_one"):
        monkeypatch.setattr(mdb, name, getattr(store, name))

    report = asyncio.run(R.import_realm(bundle, dry_run=True))
    assert report["dry_run"] is True
    assert report["realm_id"] == "demo"
    kinds = {e["kind"]: e["created"] for e in report["entries"]}
    assert kinds["prompts"] == 1
    assert kinds["generation_presets"] == 1
    assert kinds["datasets"] == 1


def test_demo_declares_the_supported_format(bundle):
    assert bundle["format"] == R._EXPORT_FORMAT


def test_demo_questions_carry_ground_truth_and_refs(bundle):
    """A question with no reference answer produces metrics nothing can explain:
    `answer_similarity` has nothing to compare against. The first run over the
    demo must show meaningful numbers, or the demo teaches people to read
    dashes."""
    questions = bundle["datasets"][0]["questions"]
    assert len(questions) >= 10
    for q in questions:
        assert q["ground_truth"].strip(), q["id"]
        assert q["question"].endswith("?"), q["id"]


def test_demo_covers_both_answerable_and_out_of_scope(bundle):
    """`correct_refusal` is one of the five metrics the demo realm puts on its
    Overview page, and it can only be counted over questions the corpus does not
    answer. A dataset where every question is answerable shows that metric as a
    dash on the first screen anybody sees.

    An empty `article_refs` is what marks those: `core/eval/answerability.py`
    classifies such a question `out_of_scope`, which is what makes declining the
    right behaviour and not a miss."""
    questions = bundle["datasets"][0]["questions"]
    answerable = [q for q in questions if q["article_refs"]]
    out_of_scope = [q for q in questions if not q["article_refs"]]
    assert len(answerable) >= 10
    assert len(out_of_scope) >= 2


def test_demo_refs_are_ids_the_corpus_can_produce(bundle):
    """The reference form has to match what ingestion actually emits, or every
    retrieval metric reads 0.0 while looking perfectly well-formed.

    This is the defect the demo shipped with: refs written as prose ("section
    2.1") can never equal a ref id, so `retrieval_recall_at_k` was 0.0 for all
    twelve questions and `answerability` classified every one of them
    `uncovered`, dropping them from the retrieval count entirely. The demo was
    teaching its first-time reader to read zeroes.

    `extract_ref_id` builds "{source_code}/{article_no}", where
    `services/ingestion/cli.py` takes `source_code` from the parent directory
    name and `article_no` from a numeric filename stem. So each ref must name a
    file that exists in `corpus/demo_handbook/`."""
    corpus = pathlib.Path(__file__).parents[2] / "corpus" / "demo_handbook"
    available = {f"{corpus.name}/{p.stem}" for p in corpus.glob("*.md")}
    assert available, "demo corpus is missing"

    refs = {ref for q in bundle["datasets"][0]["questions"] for ref in q["article_refs"]}
    assert refs <= available, sorted(refs - available)


def test_demo_prompt_asks_for_a_citation_form_the_code_substitutes(bundle):
    """`core/citation.py#substitute_fragment_markers` replaces "Fragment N" in a
    generated answer with the real section label of the Nth context entry, and
    it matches that one form. A prompt asking for any other marker leaves the
    answer carrying a number that means nothing on its own, and
    `computed_citations` empty.

    The demo shipped asking for "[section N]", so on the platform's own showcase
    the citation machinery never fired."""
    from core.citation import substitute_fragment_markers

    # The template carries the marker with a literal "N" where the model will
    # write a number, so the check substitutes one and then asserts the
    # rewriting actually happens.
    template = bundle["prompts"][0]["template"].replace("N)", "1)")
    assert substitute_fragment_markers(template, _REFS) != template, (
        "the demo prompt asks for a citation marker core/citation.py does not rewrite"
    )


def test_demo_preset_asks_for_the_shape_the_generator_parses(bundle):
    """A generation preset replaces the built-in template wholesale, so it has
    to ask for the JSON object `_parse_llm_json` reads back.

    The demo shipped a preset asking for a plain sentence. The parser found no
    object in the reply and every draft failed as "unparseable_response":
    twenty out of twenty on the realm the platform ships to be tried first.

    Checked against the parser's own list of required keys, so a preset
    reworded later still has to carry them, and against the literal example
    object the template shows the model to imitate."""
    import json
    import re

    from services.api_gateway.routers.generation import _parse_llm_json

    template = bundle["generation_presets"][0]["template"]
    # The example object, not the `{question_type}` placeholders that also use
    # braces: the one brace span that parses as JSON.
    examples = [
        m.group() for m in re.finditer(r"\{[^{}]*\}", template)
        if _is_json_object(m.group().replace("...", "x"))
    ]
    assert examples, "the preset shows the model no JSON object to imitate"
    shape = json.loads(examples[-1].replace("...", "x"))
    assert set(shape) == {"question", "reference_answer"}, shape
    assert _parse_llm_json(examples[-1].replace("...", "x")) is not None


def _is_json_object(text: str) -> bool:
    import json
    try:
        return isinstance(json.loads(text), dict)
    except ValueError:
        return False


def test_demo_bundle_matches_its_generator(bundle, monkeypatch):
    """The checked-in bundle is derived from `eval/golden/handbook.v1.fast.jsonl`
    by `tools/seed_demo.build_bundle`, and regenerating it is an opt-in flag
    (`--write-bundle`). A derived file that nothing compares drifts, and this one
    did: the dataset moved to the English handbook while the bundle still held
    the twelve Russian questions it was born with, so the button on the welcome
    screen and the corpus the installer ingests described different demos.

    `build_bundle` reads OLLAMA_MODEL to point a live realm at whatever model is
    installed. The file in the repository is the default form, so the comparison
    unsets it."""
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    from tools.seed_demo import build_bundle, load_questions

    assert bundle == build_bundle(load_questions()), (
        "ui/public/demo.realm.json is stale: run `python3 -m tools.seed_demo --write-bundle`"
    )


def test_demo_has_no_secrets(bundle):
    """The demo lives in the repository and is served over HTTP to anybody who
    opens the platform. A password inside it is a password published with the
    build."""
    for resource in bundle["realm"]["resources"]:
        for key, value in resource.items():
            if R._SENSITIVE_KEY.search(key):
                assert not value, f"{resource['type']}.{key}"
    assert bundle["masked_fields"] == []


def test_demo_carries_corpus_by_reference_only(bundle):
    """Corpus contents never travel in an export, neither in the demo nor in a
    working realm's transfer."""
    assert bundle["corpora"]
    for corpus in bundle["corpora"]:
        assert set(corpus) <= {"corpus_id", "description"}


def test_demo_names_no_real_domain(bundle):
    """The demo is the first thing anybody sees. A real subject area inside it
    teaches that the platform is about that area; the guide has already been
    rewritten to name none."""
    text = json.dumps(bundle, ensure_ascii=False).lower()
    # Markers of one specific subject area only, rather than any word that
    # happens to appear in a working realm. "Equipment acceptance" is ordinary
    # procurement wording rather than a fingerprint of anything; the first version
    # of this list banned it and caught our own data instead of somebody
    # else's domain.
    # The words stay Russian on purpose: a leak from the working realms would
    # arrive in Russian, so those are the strings worth looking for.
    for word in ("статья", "нпа", "кодекс", "исковой", "медицин", "пациент"):
        assert word not in text, word
