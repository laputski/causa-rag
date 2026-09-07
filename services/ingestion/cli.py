"""Ingestion CLI: ingest <path>

Reads txt/md files, chunks with selected strategy, embeds, and upserts into
Qdrant (dense) and OpenSearch (sparse) — each strategy gets its own namespace.

Usage:
    python -m services.ingestion.cli ingest ./docs/ --strategy fixed
    python -m services.ingestion.cli ingest ./docs/ --strategy structure_aware
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from adapters.bge_m3 import BgeM3Embedder
from core.chunking.fixed import FixedChunkingStrategy
from core.chunking.post_conditions import unmet
from core.chunking.structure_aware import StructureAwareChunkingStrategy
from core.interfaces import ChunkingStrategy
from core.models import Document

log = structlog.get_logger()

SUPPORTED_EXTENSIONS = {".txt", ".md"}

# Compound article numbers ("4.7", "16.9", ...) aren't caught by
# .isdigit() — see _read_file.
_ARTICLE_NO_RE = re.compile(r"^\d+(\.\d+)*$")

_STRATEGIES: dict[str, ChunkingStrategy] = {
    "fixed": FixedChunkingStrategy(),
    "structure_aware": StructureAwareChunkingStrategy(),
}


def _read_file(path: Path, structure_parser_fn: Any | None = None) -> Document:
    content = path.read_text(encoding="utf-8")
    structure = structure_parser_fn(content) if structure_parser_fn is not None else None
    # Eval Measurement Trustworthiness, Phase 0 — the parent directory name
    # and numeric filename stem are the source/article identity used to
    # disambiguate retrieval matches across corpora that share article
    # numbers (e.g. an article 5 exists in multiple codes). Generic by
    # construction: any corpus laid out as <source_code>/<article_no>.txt
    # gets this for free, nothing legal-specific here.
    source_code = path.parent.name
    article_no = path.stem if _ARTICLE_NO_RE.match(path.stem) else None
    return Document(
        source=str(path),
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        structure=structure,
        metadata={
            "filename": path.name,
            "suffix": path.suffix,
            "source_code": source_code,
            "article_no": article_no,
        },
    )


def _collect_files(root: Path, exclude: set[str] | None = None) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix in SUPPORTED_EXTENSIONS else []
    return sorted(
        p for p in root.rglob("*")
        if p.suffix in SUPPORTED_EXTENSIONS and (exclude is None or p.name not in exclude)
    )


def ingest(
    source: Path,
    strategy_id: str = "fixed",
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    opensearch_host: str = "localhost",
    opensearch_port: int = 9200,
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
    chunk_size: int = 512,
    overlap: int = 64,
    use_opensearch: bool = True,
    exclude: set[str] | None = None,
    corpus_id: str = "default",
    structure: str | None = None,
    realm_id: str | None = None,
    # The analyser the sparse index is built with. Only matters at creation:
    # an index carries its analyser for life and queries inherit it.
    language: str = "ru_be",
) -> dict[str, Any]:
    files = _collect_files(source, exclude=exclude)
    if not files:
        log.warning("ingest.no_files", path=str(source))
        # The same keys a real load returns, and that is the whole point of
        # listing them. This branch used to return three of them, one under a
        # name nothing else uses, and the caller printing the cache ratio died
        # with a KeyError on a path that had simply matched no files. Found by
        # trying to stage the catalogue's entry for an index holding nothing:
        # the failure could not be reached, because the loader crashed before
        # it could happen.
        return {
            "files": 0, "chunks": 0, "hits": 0, "misses": 0, "hit_ratio": 0.0, "size": 0,
            "qdrant_collection": None, "opensearch_index": None, "manifest": {},
        }

    # Structure parser — resolved once via the domain-pack
    # registry by "<pack_id>/<parser_id>" (e.g. "manuals/manual_section"), never
    # imported directly. Loaded once, not per-file.
    structure_parser_fn = None
    if structure:
        from core.domain.loader import load_pack
        from core.registry import registry

        pack_id, _, parser_id = structure.partition("/")
        if not parser_id:
            raise ValueError(
                f"--structure must be '<pack_id>/<parser_id>' (e.g. manuals/manual_section), got {structure!r}"
            )
        load_pack(pack_id, registry, {})
        structure_parser_fn = registry.resolve("structure_parser", parser_id)

    # Select chunking strategy — namespace-per-strategy
    if strategy_id == "fixed":
        chunker: ChunkingStrategy = FixedChunkingStrategy(chunk_size=chunk_size, overlap=overlap)
    elif strategy_id == "structure_aware":
        chunker = StructureAwareChunkingStrategy(max_chunk_size=chunk_size, fallback_overlap=overlap)
    else:
        raise ValueError(f"Unknown strategy: {strategy_id!r}. Available: {list(_STRATEGIES)}")

    embedder = BgeM3Embedder()

    # Dense index — namespace per strategy
    from adapters.qdrant import QdrantRetriever
    qdrant = QdrantRetriever(
        host=qdrant_host,
        port=qdrant_port,
        strategy_id=strategy_id,
        embedder_id=embedder.embedder_id,
        corpus_id=corpus_id,
        realm_id=realm_id,
    )

    # Sparse index — namespace per strategy
    opensearch = None
    if use_opensearch:
        try:
            from adapters.opensearch import AnalyzerMismatch, OpenSearchRetriever
            opensearch = OpenSearchRetriever(
                host=opensearch_host,
                port=opensearch_port,
                strategy_id=strategy_id,
                corpus_id=corpus_id,
                realm_id=realm_id,
                language=language,
            )
        except AnalyzerMismatch:
            # Let through deliberately. Everything else here is "OpenSearch is
            # not up", which is a reason to go on without a sparse index; this
            # one is "the index would stem the text by the wrong language's
            # rules", and going on would produce numbers that mean nothing.
            raise
        except Exception as exc:
            log.warning("ingest.opensearch_unavailable", error=str(exc))

    # Optional graph index — built from chunks when enabled.
    graph = None
    if os.getenv("USE_GRAPH", "").lower() == "true":
        try:
            from adapters.neo4j_graph import Neo4jGraphRetriever
            candidate = Neo4jGraphRetriever(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
            if candidate.is_available() and candidate.verify():
                graph = candidate
                candidate.ensure_indexes()
            else:
                log.warning("ingest.graph_unavailable", reason="neo4j_unreachable")
        except Exception as exc:
            log.warning("ingest.graph_unavailable", error=str(exc))

    total_chunks = 0
    # The fingerprint of what was loaded, kept this time. Every
    # document's content hash was computed at read time and dropped on the
    # floor: the field existed, one line assigned it, and nothing stored it.
    # Three catalogue entries were uncatchable for exactly that, because
    # "this corpus was reindexed" and "this is a different corpus" are the
    # same absence of a record.
    document_hashes: list[str] = []
    # Kept so the strategy can be asked, once, whether it did what its name
    # says. Per document the question is unanswerable: one flat file among
    # forty is a fact about that file.
    all_chunks: list[Any] = []
    for path in files:
        doc = _read_file(path, structure_parser_fn=structure_parser_fn)
        chunks = chunker.chunk(doc)
        if not chunks:
            continue
        document_hashes.append(doc.content_hash)
        all_chunks.extend(chunks)

        # Dense
        texts = [c.text for c in chunks]
        vectors = embedder.embed(texts)
        qdrant.upsert(chunks, vectors)

        # Sparse
        if opensearch:
            opensearch.index_chunks(chunks)

        # Graph — per-file node/adjacency upsert only; cross-doc RELATED edges
        # are linked once after the loop (avoids O(files × n²) re-scans).
        if graph is not None:
            try:
                graph.add_chunks(chunks)
            except Exception as exc:
                log.warning("ingest.graph_build_failed", file=path.name, error=str(exc))

        total_chunks += len(chunks)
        log.info("ingest.file_done", file=path.name, chunks=len(chunks), strategy=strategy_id)

    # Shared-keyword links: built once over the whole graph after ingest.
    if graph is not None:
        try:
            graph.link_related()
        except Exception as exc:
            log.warning("ingest.graph_link_failed", error=str(exc))
        # Embedding-similarity links — second, semantic signal alongside the
        # lexical RELATED edges above (kept as a distinct relationship type,
        # see adapters/neo4j_graph.py:link_semantic). On by default for every
        # graph-enabled ingest, not an opt-in maintenance step.
        try:
            graph.link_semantic(qdrant)
        except Exception as exc:
            log.warning("ingest.graph_semantic_link_failed", error=str(exc))

    stats = embedder.cache_stats()
    log.info(
        "ingest.done",
        files=len(files),
        total_chunks=total_chunks,
        strategy=strategy_id,
        **stats,
    )
    # Surfaced so a caller (services/api_gateway/routers/corpus.py's
    # /corpus/ingest) can register the *actual* physical collection/index name
    # in the `corpora` registry without re-deriving it independently, which
    # would risk drifting from adapters/qdrant.py#_collection_name's own logic.
    return {
        "files": len(files), "chunks": total_chunks, **stats,
        "qdrant_collection": qdrant._collection,
        "opensearch_index": opensearch._index if opensearch else None,
        "manifest": _manifest(embedder, strategy_id, chunk_size, overlap,
                              document_hashes, total_chunks, all_chunks),
    }


def _manifest(
    embedder: Any, strategy_id: str, chunk_size: int, overlap: int,
    document_hashes: list[str], chunk_count: int, chunks: list[Any] | None = None,
) -> dict[str, Any]:
    """What this index was built from and built by.

    Recorded because three questions have no answer without it, and each of
    them looks from the outside like a silence and not like a gap: whether the
    documents changed since the index was built, whether the model that
    indexed them is the model that queries them, and whether the vectors came
    from a model at all.

    That last one is the sharpest. The embedder's id and version are class
    constants, identical for the working model and for the stub that hashes
    text into a vector, so a corpus indexed by either is described the same
    way afterwards. Asking the embedder here, where the indexing happens, is
    the only place the answer exists.

    The digest is over the documents' own hashes, sorted, so it is a property
    of the set and not of the order the files were walked in.
    """
    return {
        "embedder_id": getattr(embedder, "embedder_id", ""),
        "embedder_version": getattr(embedder, "version", ""),
        "embedder_is_real_model": bool(getattr(embedder, "is_real_model", False)),
        "chunking_strategy": strategy_id,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "document_count": len(document_hashes),
        "chunk_count": chunk_count,
        "documents_digest": hashlib.sha256(
            "".join(sorted(document_hashes)).encode()
        ).hexdigest(),
        # What the strategy promised about its own output and did not keep.
        # A strategy named for structure that produced none has done exactly
        # what the plain fixed-window strategy does, under another name, and
        # every other trace of that load looks correct.
        "post_conditions_unmet": unmet(strategy_id, chunks or [], chunk_size),
        "loaded_at": datetime.now(UTC).isoformat(),
    }


def _register_in_the_realm(realm_id: str, corpus_id: str, result: dict[str, Any]) -> str:
    """Tell the realm about the corpus that was just loaded into it.

    A load names a realm and, until now, the realm learned nothing: the index
    existed, every metric could be computed against it, and it appeared in no
    list on any screen. Found while writing a manual check of the proving
    ground, where a deliberately damaged corpus was loaded and then could not
    be selected to look at, which is half of what that realm is for.

    The registry record is what the interface lists, and the helper that
    writes it declares itself idempotent and safe to call on every successful
    load. It is called here for the same reason the gateway's own ingest route
    calls it: whoever names a realm means it.

    Failure to register is reported and does not fail the load. The documents
    are in the index either way, and a load that succeeded is not undone by a
    database that was unreachable a second later.
    """
    import asyncio

    from services.api_gateway.routers import corpus as corpus_router

    manifest = result.get("manifest") or {}
    backends: dict[str, dict[str, Any]] = {}
    if result.get("qdrant_collection"):
        # The embedder is read from what indexed, never written down here.
        # It used to be the literal "bge_m3", which was true of every load
        # anybody had run and would have gone on being recorded whatever
        # indexed the next one.
        backends["qdrant"] = {
            "collection": result["qdrant_collection"],
            "embedder_id": manifest.get("embedder_id", ""),
        }
    if result.get("opensearch_index"):
        backends["opensearch"] = {"index": result["opensearch_index"]}
    try:
        asyncio.run(corpus_router._register_corpus(
            realm_id=realm_id, corpus_id=corpus_id,
            storage_type="hybrid" if "opensearch" in backends else "dense_only",
            backends=backends, owner="platform", manifest=manifest,
        ))
    except Exception as exc:
        return f"Loaded, and not registered in realm {realm_id!r}: {exc}"
    return f"Registered as corpus {corpus_id!r} of realm {realm_id!r}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into RAG platform")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("ingest", help="Ingest files")
    p.add_argument("path", type=Path)
    p.add_argument("--strategy", default="fixed", choices=list(_STRATEGIES))
    p.add_argument("--qdrant-host", default="localhost")
    p.add_argument("--qdrant-port", type=int, default=6333)
    p.add_argument("--opensearch-host", default="localhost")
    p.add_argument("--opensearch-port", type=int, default=9200)
    p.add_argument("--neo4j-uri", default=None, help="Neo4j bolt URI (default: env NEO4J_URI or bolt://localhost:7687)")
    p.add_argument("--neo4j-user", default=None, help="Neo4j username (default: env NEO4J_USER or neo4j)")
    p.add_argument("--neo4j-password", default=None, help="Neo4j password (default: env NEO4J_PASSWORD)")
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--no-opensearch", action="store_true")
    p.add_argument("--corpus-id", default="default", help="Multi-corpus namespace")
    p.add_argument("--language", default="ru_be", metavar="CODE",
                   help="Analyser language for the sparse index. Takes an ISO 639-1 "
                        "code (ar, de, it, pt, ...), a Lucene analyser name (arabic, "
                        "italian, ...), 'standard' for no stemming, or 'ru_be' (the "
                        "default: Russian stemming with Belarusian stop words). "
                        "Applies only when the index is created; an existing one keeps "
                        "what it was built with")
    p.add_argument("--realm-id", default=None, help="Realm namespace — isolates collection/index names when multiple Realms share one physical Qdrant/OpenSearch instance")
    p.add_argument("--exclude", nargs="*", default=["full.txt"], metavar="FILENAME",
                   help="Filenames to skip (default: full.txt)")
    p.add_argument("--structure", default=None, metavar="PACK/PARSER",
                   help="Domain-pack structure parser as '<pack_id>/<parser_id>' "
                        "(e.g. manuals/manual_section) to populate doc.structure for "
                        "structure_aware chunking (default: none -> flat split)")

    args = parser.parse_args()
    if args.command == "ingest":
        result = ingest(
            source=args.path,
            strategy_id=args.strategy,
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
            opensearch_host=args.opensearch_host,
            opensearch_port=args.opensearch_port,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            use_opensearch=not args.no_opensearch,
            exclude=set(args.exclude) if args.exclude else None,
            corpus_id=args.corpus_id,
            structure=args.structure,
            language=args.language,
            realm_id=args.realm_id,
        )
        if not result["files"]:
            # Said plainly and refused, because the quiet version of this is a
            # catalogue entry: an index holding nothing answers every question
            # with nothing, every retrieval metric comes back at zero, and zero
            # is also what a catastrophically bad retriever produces.
            print(f"No documents found in {args.path}. Nothing was indexed.")
            sys.exit(1)
        print(f"Ingested {result['chunks']} chunks | cache hit ratio: {result['hit_ratio']:.2%}")
        if args.realm_id:
            print(_register_in_the_realm(args.realm_id, args.corpus_id, result))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
