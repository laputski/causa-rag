"""Fetch a MIRACL language slice and lay it out the way ingestion expects.

The layout is the whole trick. Retrieval metrics
(core/eval/retrieval_metrics.py) compare `{source_code}/{article_no}`
strings, where services/ingestion/cli.py takes source_code from a file's
parent directory and article_no from its numeric stem. A MIRACL passage is
identified as `151236#1`, which is that same pair written with a different
separator, so writing it to `151236/1.txt` makes the golden set's
`article_refs` a straight copy of the qrels with nothing to translate.

The dataset lives behind a loading script on HuggingFace, and the viewer
answers 501 for it ("runs arbitrary Python code"), so nothing here goes
through `datasets`. Topics and qrels are plain TSV; the corpus is gzipped
JSONL shards, streamed and filtered rather than kept, because English alone
is 5 GB compressed and only about eight thousand of its passages are judged.

Usage:
    python3 -m eval.miracl.fetch --lang ar --limit 200 --dry-run
    python3 -m eval.miracl.fetch --lang ar
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import ssl
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_HF = "https://huggingface.co"
_TOPICS = _HF + "/datasets/miracl/miracl/resolve/main/miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-{split}.tsv"
_QRELS = _HF + "/datasets/miracl/miracl/resolve/main/miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-{split}.tsv"
_TREE = _HF + "/api/datasets/miracl/miracl-corpus/tree/main/miracl-corpus-v1.0-{lang}"
_SHARD = _HF + "/datasets/miracl/miracl-corpus/resolve/main/{path}"

# Only `dev` has public judgements. test-a and test-b withhold theirs, so a
# run against them would score every configuration at zero and look like a
# broken pipeline rather than a missing answer key.
SPLIT = "dev"

# The same shape services/ingestion/cli.py requires of a filename stem
# before it will record an article_no at all. A docid that fails it would
# be ingested with article_no=None and silently drop out of every metric.
_NUMERIC = re.compile(r"^\d+(\.\d+)*$")

_UA = {"User-Agent": "causa-rag/0.1 (+https://laputski.ai)"}


def _ssl_context() -> ssl.SSLContext:
    """Verified TLS, with certifi when the interpreter has no root store.

    A python.org build on macOS ships without one and fails every HTTPS
    request here. Turning verification off would be the short way out and
    is not taken: this downloads data that the published report's numbers
    rest on.
    """
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _open(url: str, timeout: int = 120) -> Any:
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=_UA), timeout=timeout, context=_ssl_context(),
    )


def _get_text(url: str) -> str:
    with _open(url) as resp:
        return str(resp.read().decode("utf-8"))


def _get_json(url: str) -> object:
    with _open(url, timeout=60) as resp:
        return json.load(resp)


@dataclass
class Slice:
    """One language's dev split, before anything is written to disk."""

    lang: str
    questions: dict[str, str]                  # qid -> query text
    positives: dict[str, list[str]]            # qid -> relevant docids
    judged: set[str]                           # every docid a human looked at

    @property
    def n_relevant(self) -> int:
        return sum(len(v) for v in self.positives.values())


def read_topics(lang: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in _get_text(_TOPICS.format(lang=lang, split=SPLIT)).splitlines():
        if not line.strip():
            continue
        qid, _, query = line.partition("\t")
        out[qid.strip()] = query.strip()
    return out


def read_qrels(lang: str) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Relevant docids per query, and every docid judged for that query.

    Both are kept, per query, and they are not the same set. The judged
    docids are the retrieval pool: they carry the passages a human read and
    rejected, which are the only distractors this report can afford. A pool
    built from the relevant ones alone is a pool where every document is an
    answer, and every configuration scores near the top of it.

    Per query rather than flat, because --limit selects questions and the
    pool has to narrow with them; a flat set would have to be thrown away
    and the small run would lose exactly its distractors.
    """
    positives: dict[str, list[str]] = defaultdict(list)
    judged: dict[str, list[str]] = defaultdict(list)
    for line in _get_text(_QRELS.format(lang=lang, split=SPLIT)).splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        qid, _, docid, rel = parts[0], parts[1], parts[2], parts[3]
        judged[qid].append(docid)
        if int(rel) > 0:
            positives[qid].append(docid)
    return dict(positives), dict(judged)


def split_docid(docid: str) -> tuple[str, str]:
    wiki_id, _, passage = docid.partition("#")
    return wiki_id, passage


def unlayoutable(docids: set[str]) -> list[str]:
    """Docids that would not survive the round trip through the filesystem.

    Checked before a single byte of corpus is downloaded, and fatal rather
    than skipped. A dropped judged passage does not announce itself later:
    it shrinks a recall denominator, and the configuration that needed it
    simply scores lower for no visible reason.
    """
    bad = []
    for docid in docids:
        wiki_id, passage = split_docid(docid)
        if not wiki_id or not passage or not _NUMERIC.match(wiki_id) or not _NUMERIC.match(passage):
            bad.append(docid)
    return sorted(bad)


def build_slice(lang: str, limit: int | None = None) -> Slice:
    topics = read_topics(lang)
    positives, judged = read_qrels(lang)

    # Questions with no positive judgement cannot be scored; they are
    # dropped here rather than counted as failures later.
    qids = sorted(q for q in topics if positives.get(q))
    if limit is not None:
        qids = qids[:limit]

    return Slice(
        lang=lang,
        questions={q: topics[q] for q in qids},
        positives={q: positives[q] for q in qids},
        judged={d for q in qids for d in judged.get(q, ())},
    )


def shard_paths(lang: str) -> list[str]:
    tree = _get_json(_TREE.format(lang=lang))
    assert isinstance(tree, list)
    return sorted(e["path"] for e in tree if str(e.get("path", "")).endswith(".jsonl.gz"))


def stream_shard(path: str) -> Iterator[dict[str, Any]]:
    with _open(_SHARD.format(path=path)) as resp, gzip.open(resp, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_passage(root: Path, docid: str, title: str, text: str) -> None:
    wiki_id, passage = split_docid(docid)
    target = root / wiki_id
    target.mkdir(parents=True, exist_ok=True)
    # Title first, blank line, then the passage. MIRACL stores them apart,
    # and a passage that has lost its article title loses the one term most
    # likely to match a query naming the subject.
    body = f"{title}\n\n{text}\n" if title else f"{text}\n"
    (target / f"{passage}.txt").write_text(body, encoding="utf-8")


def fetch_corpus(
    sl: Slice, root: Path, log: Callable[[str], None] = print,
) -> tuple[int, set[str]]:
    """Stream every shard, keep the judged passages, discard the rest.

    Shards are not cached. The alternative is holding 7 GB across three
    languages to re-extract about 46 000 passages from it, and the passages
    themselves are what a rerun needs.
    """
    wanted = set(sl.judged)
    found = 0
    paths = shard_paths(sl.lang)
    for i, path in enumerate(paths, 1):
        if not wanted:
            log(f"  every judged passage found, {len(paths) - i + 1} shards not downloaded")
            break
        before = len(wanted)
        for doc in stream_shard(path):
            docid = doc.get("docid", "")
            if docid in wanted:
                write_passage(root, docid, doc.get("title", ""), doc.get("text", ""))
                wanted.discard(docid)
                found += 1
        log(f"  shard {i}/{len(paths)}: +{before - len(wanted)} passages, {len(wanted)} still wanted")
    return found, wanted


def write_golden(sl: Slice, path: Path) -> None:
    """One question per line, with no ground_truth field.

    Its absence is the honest record of what MIRACL is. The runner reads
    reference_answer with a default, so answer-similarity scoring simply
    has nothing to compare and reports nothing, instead of comparing
    against an empty string and reporting a confident zero.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for qid in sorted(sl.questions):
            f.write(json.dumps({
                "id": qid,
                "question": sl.questions[qid],
                "question_type": "open",
                "mode": "retrieval",
                "article_refs": sorted(
                    "/".join(split_docid(d)) for d in sl.positives[qid]
                ),
            }, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, help="MIRACL language code, e.g. ar, ru, en")
    ap.add_argument("--limit", type=int, default=None, help="keep only the first N scorable questions")
    ap.add_argument("--dry-run", action="store_true", help="report what would be fetched, write nothing")
    ap.add_argument("--corpus-root", type=Path, default=Path("corpus"))
    ap.add_argument("--golden-root", type=Path, default=Path("eval/golden"))
    args = ap.parse_args(argv)

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    log(f"reading topics and qrels for {args.lang} ({SPLIT})")
    try:
        sl = build_slice(args.lang, args.limit)
    except urllib.error.HTTPError as exc:
        log(f"cannot read {args.lang}: HTTP {exc.code}. Is that a MIRACL language code?")
        return 2

    bad = unlayoutable(sl.judged)
    if bad:
        log(f"{len(bad)} judged docids do not fit the {{wiki_id}}/{{passage}} layout, "
            f"first few: {bad[:5]}")
        log("refusing to build a golden set that would quietly omit them")
        return 3

    log(f"  {len(sl.questions)} scorable questions, {sl.n_relevant} relevant passages, "
        f"{len(sl.judged)} in the pool")

    corpus_root = args.corpus_root / f"miracl-{args.lang}"
    golden_path = args.golden_root / f"miracl-{args.lang}.v1.fast.jsonl"

    if args.dry_run:
        log(f"dry run: would write {len(sl.judged)} files under {corpus_root}")
        log(f"dry run: would write {golden_path}")
        return 0

    corpus_root.mkdir(parents=True, exist_ok=True)
    args.golden_root.mkdir(parents=True, exist_ok=True)

    log(f"streaming the {args.lang} corpus, keeping {len(sl.judged)} passages")
    found, missing = fetch_corpus(sl, corpus_root, log)
    log(f"  wrote {found} passages under {corpus_root}")
    if missing:
        # The pool is the denominator of every recall number in the report,
        # so a hole in it is reported, never rounded away.
        log(f"  {len(missing)} judged passages were not in the corpus, first few: "
            f"{sorted(missing)[:5]}")

    write_golden(sl, golden_path)
    log(f"  wrote {golden_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
