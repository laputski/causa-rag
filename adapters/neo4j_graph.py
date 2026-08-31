"""Neo4j-backed GraphRetriever — functional GraphRAG.

Builds a chunk-level knowledge graph at ingest (nodes = chunks, edges = document
adjacency + shared-term links) and retrieves via keyword-seed + n-hop Cypher
traversal. Requires the optional ``[graph]`` extra (the ``neo4j`` driver) and a
running Neo4j instance (see deploy/compose). When either is unavailable the
caller should fall back to ``LightRagRetrieverStub`` — see ``is_available``.
"""
from __future__ import annotations

import os
import re
from typing import Any

from adapters.lightrag import GraphNode, GraphResult
from core.models import Chunk, ScoredChunk

_WORD = re.compile(r"\w{4,}", re.UNICODE)


def _keywords(text: str, limit: int = 8) -> list[str]:
    """Cheap keyword extraction — longest distinct lowercased tokens."""
    seen: dict[str, None] = {}
    for m in _WORD.findall(text.lower()):
        seen.setdefault(m, None)
    return list(seen)[:limit]


class GraphUnavailable(RuntimeError):
    """Raised when the Neo4j driver or server is not reachable."""


class Neo4jGraphRetriever:
    """Production GraphRetriever over Neo4j."""

    retriever_id = "lightrag"  # same id as the production LightRAG path (interchangeable)

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        connection_timeout: float | None = None,
    ) -> None:
        # Annotated: os.getenv with a default never returns None, and the
        # driver refuses anything else, but mypy widens the `or` to str | None
        # and the resulting error hid behind an unread check for weeks.
        self._uri: str = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self._user = user or os.getenv("NEO4J_USER", "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD", "ragplatform")
        # The driver's own default is 30s, which is right for a pipeline that
        # is going to run for minutes and wrong for a health check somebody is
        # watching. Callers that ask a yes/no question pass their own bound.
        self._connection_timeout = connection_timeout
        self._driver: Any | None = None

    @staticmethod
    def is_available() -> bool:
        """Whether the optional ``neo4j`` driver is importable — **not**
        whether a server is running.

        The distinction is the whole point and the name hides it, so every
        caller pairs this with ``verify()`` (see services/ingestion/cli.py and
        tests/integration/conftest.py). One did not: the Realm resource test
        (``services/api_gateway/routers/realms.py#_ping_resource``) called this
        alone and therefore reported ``{"status": "ok"}`` for a Neo4j that was
        not running — found live on 18.08.2026, with port 7687 closed and no
        container for that Realm at all. A health screen built on a check that
        answers a different question than the one asked reports honestly and
        says something false.

        Kept as it is rather than made to connect: the fall-back-to-stub path
        genuinely wants "is the driver installed", cheaply and without a
        socket. What changed is the caller.
        """
        try:
            import neo4j  # noqa: F401
            return True
        except Exception:
            return False

    def _connect(self) -> Any:
        if self._driver is None:
            try:
                from neo4j import GraphDatabase
            except Exception as exc:
                raise GraphUnavailable("neo4j driver not installed ('.[graph]')") from exc
            try:
                kwargs: dict[str, Any] = {"auth": (self._user, self._password)}
                if self._connection_timeout is not None:
                    kwargs["connection_timeout"] = self._connection_timeout
                self._driver = GraphDatabase.driver(self._uri, **kwargs)
                self._driver.verify_connectivity()
            except Exception as exc:
                self._driver = None
                raise GraphUnavailable(f"Neo4j unreachable at {self._uri}: {exc}") from exc
        return self._driver

    def verify(self) -> bool:
        try:
            self._connect()
            return True
        except GraphUnavailable:
            return False

    def close(self) -> None:
        """Releases the driver's connection pool. Safe to call even if
        `_connect()` was never reached (no-op)."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    # ── graph construction (called at ingest) ─────────────────────────────────

    def clear(self) -> None:
        """Wipe the entire graph (all nodes + relationships).

        The graph has no corpus_id partitioning — link_related() matches
        across all :Chunk nodes regardless of which ingest wrote them. Call
        this before ingesting a new corpus to avoid cross-corpus RELATED
        edges between unrelated documents.

        Batched via apoc.periodic.iterate — a single `MATCH (n) DETACH
        DELETE n` transaction on a graph this size (35k+ nodes, ~1M edges)
        hit Neo4j's dbms.memory.transaction.total.max and raised
        TransientError (same class of issue link_related() already works
        around, see its docstring) — found live: the un-batched version
        silently failed (session.run()'s lazy Result was never consumed,
        so the error never surfaced) and left the graph untouched while
        the caller believed it had been cleared.
        """
        driver = self._connect()
        with driver.session() as session:
            session.run(
                "CALL apoc.periodic.iterate("
                "'MATCH (n) RETURN n', "
                "'DETACH DELETE n', "
                "{batchSize: 2000})"
            ).consume()

    def ensure_indexes(self) -> None:
        """Create the indexes graph build/retrieval rely on (idempotent).

        Index on :Chunk(chunk_id) speeds the per-chunk MERGE and NEXT lookups;
        index on :Chunk(keywords) speeds the shared-keyword bucketing in
        ``link_related`` and the keyword-seed match in ``retrieve_graph``.
        """
        driver = self._connect()
        with driver.session() as session:
            session.run(
                "CREATE INDEX chunk_id_idx IF NOT EXISTS FOR (n:Chunk) ON (n.chunk_id)"
            )
            session.run(
                "CREATE INDEX chunk_keywords_idx IF NOT EXISTS FOR (n:Chunk) ON (n.keywords)"
            )

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Upsert chunk nodes + per-document adjacency (NEXT) edges.

        Cheap and local — touches only the given chunks, so it can be called
        once per file inside the ingest loop. Cross-document shared-keyword
        (RELATED) edges are deferred to ``link_related``, run once at the end.
        Returns the number of chunks written.
        """
        driver = self._connect()
        with driver.session() as session:
            for c in chunks:
                session.run(
                    "MERGE (n:Chunk {chunk_id: $cid}) "
                    "SET n.doc_id=$doc, n.text=$text, n.path=$path, n.keywords=$kw",
                    cid=c.chunk_id, doc=c.doc_id, text=c.text,
                    path=c.structural_path, kw=_keywords(c.text),
                )
            # adjacency edges: consecutive chunks within a document
            by_doc: dict[str, list[Chunk]] = {}
            for c in chunks:
                by_doc.setdefault(c.doc_id, []).append(c)
            for doc_chunks in by_doc.values():
                for a, b in zip(doc_chunks, doc_chunks[1:], strict=False):
                    session.run(
                        "MATCH (x:Chunk {chunk_id:$a}),(y:Chunk {chunk_id:$b}) "
                        "MERGE (x)-[:NEXT]->(y)",
                        a=a.chunk_id, b=b.chunk_id,
                    )
        return len(chunks)

    def link_related(self, max_keyword_freq: int = 50) -> None:
        """Create shared-keyword (RELATED) edges across the whole graph, once.

        Two problems found at real corpus scale (31k chunks, 10x the
        previous corpus) that don't show up on small corpora:

        1. A single transaction bucketing ALL keywords at once exceeded
           Neo4j's dbms.memory.transaction.total.max (hard OOM). Fixed by
           batching via apoc.periodic.iterate, one keyword per committed
           transaction.
        2. Generic legal-document boilerplate words, the domain's equivalents
           of "hereinafter" or "pursuant to", appear in thousands of
           chunks across unrelated documents — bucketing on them alone
           produced 14.7M near-meaningless edges (one keyword's bucket of
           size N contributes N*(N-1)/2 pairs). Fixed by excluding keywords
           whose chunk-frequency exceeds `max_keyword_freq` BEFORE bucketing
           — chosen dynamically per corpus (frequency, not a hardcoded
           stop-word list), so it adapts to whatever vocabulary the corpus
           actually has instead of a list tuned to one snapshot.

        Call this exactly once after the whole ingest completes — never per
        file.
        """
        # max_keyword_freq is an internal int constant (never user input) —
        # interpolated directly since apoc.periodic.iterate's outer/inner
        # Cypher strings don't reliably inherit the calling query's bound
        # parameters.
        max_freq = int(max_keyword_freq)
        driver = self._connect()
        with driver.session() as session:
            session.run(
                "CALL apoc.periodic.iterate("
                "'MATCH (x:Chunk) UNWIND x.keywords AS k "
                f"WITH k, count(*) AS freq WHERE freq >= 2 AND freq <= {max_freq} "
                "RETURN k', "
                "'MATCH (a:Chunk) WHERE k IN a.keywords "
                "WITH k, collect(a) AS bucket "
                "UNWIND bucket AS a UNWIND bucket AS b "
                "WITH a, b WHERE a.chunk_id < b.chunk_id "
                "MERGE (a)-[:RELATED]->(b)', "
                "{batchSize: 25, parallel: false}"
                ")"
            )

    def link_semantic(
        self,
        qdrant: Any,
        threshold: float = 0.75,
        top_k: int = 5,
        batch_size: int = 500,
        doc_ids: list[str] | None = None,
    ) -> int:
        """Create embedding-similarity (RELATED_SEMANTIC) edges across the
        whole graph, as a second signal alongside the lexical ``RELATED``
        edges — kept as a distinct relationship type, not a replacement, so
        the two can be compared (e.g. via community detection on each
        separately) rather than one silently overriding the other.

        Reuses each chunk's embedding already stored in Qdrant at ingest
        time — ``query=chunk_id`` makes Qdrant look up that point's own
        stored vector and k-NN search with it, so this needs no new vector
        computation (same trick as ``core/eval/corpus_health.py``'s
        near-duplicate detector). Full scan, not a sample: Qdrant's HNSW
        index makes one k-NN lookup cheap, so doing this once per chunk
        (linear in corpus size) stays fast, unlike pairwise embedding
        comparison (quadratic) which this approach avoids entirely.

        Call once after ingest, like ``link_related``. Returns the number
        of edges written.

        ``doc_ids``, when given, restricts the initial chunk scan to those
        documents (Neo4j has no partitioning otherwise — see ``clear()``'s
        docstring — so an unfiltered scan always walks every ``:Chunk`` node
        ever ingested into this graph, not just the caller's own). Ingestion
        leaves this ``None`` on purpose: cross-document linking is the whole
        point of a full-corpus scan. It exists so callers that only care
        about a known subset (e.g. a test seeding a handful of chunks into a
        Neo4j instance that already holds a real corpus) don't pay for a
        Qdrant round-trip per pre-existing chunk that scan would otherwise
        drag in — found live: a 3-chunk test took 30+s because it silently
        scanned 34,813 real corpus chunks first.
        """
        driver = self._connect()
        with driver.session() as session:
            if doc_ids:
                rows = session.run(
                    "MATCH (n:Chunk) WHERE n.doc_id IN $doc_ids RETURN n.chunk_id AS cid",
                    doc_ids=doc_ids,
                )
            else:
                rows = session.run("MATCH (n:Chunk) RETURN n.chunk_id AS cid")
            chunk_ids = [r["cid"] for r in rows]

        # Qdrant always returns point IDs in canonical hyphenated UUID form
        # (query_points response), but Neo4j's chunk_id property can be the
        # same UUID WITHOUT hyphens (core/models.py:_chunk_id's 32-char
        # sha256 digest, used by structure_aware chunking) — an exact-string
        # MERGE on the literal Qdrant-returned ID silently matched nothing
        # for those chunks (real bug found live: link_semantic reported
        # writing 165269 edges, the graph had zero afterward). Normalizing
        # in Cypher (`replace(x.chunk_id, '-', '')`) fixed correctness but
        # can't use the chunk_id index — full label scan per batch made a
        # 35k-chunk graph go from ~14 min to a projected ~2.5h. Building this
        # normalized→literal lookup once in Python keeps the MERGE on the
        # original literal value (index-backed), passing the resolved
        # literal Neo4j ID as MERGE's key instead of the Qdrant-returned one.
        norm_to_literal = {cid.replace("-", ""): cid for cid in chunk_ids}

        edges: list[dict[str, Any]] = []
        seen_pairs: set[frozenset[str]] = set()
        written = 0
        for chunk_id in chunk_ids:
            try:
                hits = qdrant._client.query_points(
                    collection_name=qdrant._collection, query=chunk_id, limit=top_k + 1, with_payload=False,
                ).points
            except Exception:
                continue  # point may not exist in this Qdrant collection (e.g. different corpus_id)
            for hit in hits:
                neighbor_literal = norm_to_literal.get(str(hit.id).replace("-", ""))
                if neighbor_literal is None or neighbor_literal == chunk_id or hit.score < threshold:
                    continue
                key = frozenset((chunk_id, neighbor_literal))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                edges.append({"a": chunk_id, "b": neighbor_literal, "score": float(hit.score)})

            if len(edges) >= batch_size:
                written += self._write_semantic_batch(edges)
                edges = []
        if edges:
            written += self._write_semantic_batch(edges)
        return written

    def _write_semantic_batch(self, edges: list[dict[str, Any]]) -> int:
        driver = self._connect()
        with driver.session() as session:
            result = session.run(
                "UNWIND $edges AS e "
                "MATCH (x:Chunk {chunk_id: e.a}), (y:Chunk {chunk_id: e.b}) "
                "MERGE (x)-[r:RELATED_SEMANTIC]->(y) SET r.score = e.score "
                "RETURN count(r) AS c",
                edges=edges,
            )
            written = result.single()["c"]
        return written

    # ── community detection (GDS, diagnostic — Stage "graph viz") ──────────────
    #
    # Snapshot before adding this: 35162 :Chunk nodes, 956014 RELATED edges,
    # no GDS plugin installed. Manual Leiden/Louvain runs during planning
    # found real (non-degenerate) structure — 1746 communities, modularity
    # ~0.52 — but the large communities (the ~61 non-singleton ones, holding
    # most of the 35k nodes) mix 12-27 different source codes each, meaning
    # they're driven mostly by cross-document keyword noise, not by
    # corpus/source boundaries. The one exception (a single dominant source,
    # 96% share) was the one code in the corpus with Belarusian-language
    # headers — it separates cleanly purely because its vocabulary doesn't
    # overlap with the Russian-language majority, not because of topic.
    # These methods expose that same diagnostic (community sizes + which
    # documents end up in each) via the API instead of one-off bolt queries.
    #
    # Deliberately NOT touched: add_chunks/link_related/retrieve_graph/clear
    # — this is purely additive, doesn't change retrieval behavior.

    _COMMUNITY_ALGORITHMS = ("leiden", "louvain")
    # "lexical" (shared-keyword) vs "semantic" (embedding
    # cosine) — two independent signals kept as distinct
    # relationship types so clustering can be compared between them, not
    # just one replacing the other (see link_semantic's docstring).
    _COMMUNITY_EDGE_TYPES = {"lexical": "RELATED", "semantic": "RELATED_SEMANTIC"}

    def project_graph(self, graph_name: str = "chunks", edge_type: str = "lexical") -> tuple[int, int]:
        """(Re)creates the named GDS in-memory graph projection.

        Undirected — gds.leiden.* requires it ("works only with undirected
        graphs"); gds.louvain.* accepts either, so undirected is used for
        both to keep one projection shared across algorithms. Idempotent:
        drops any existing projection under the same name first (GDS
        catalog entries don't survive a server restart anyway, and a stale
        projection from before new chunks were added would silently miss
        them).
        """
        if edge_type not in self._COMMUNITY_EDGE_TYPES:
            raise ValueError(
                f"unsupported edge_type {edge_type!r}, expected one of {tuple(self._COMMUNITY_EDGE_TYPES)}"
            )
        rel_type = self._COMMUNITY_EDGE_TYPES[edge_type]
        driver = self._connect()
        with driver.session() as session:
            session.run("CALL gds.graph.drop($name, false)", name=graph_name)
            row = session.run(
                f"CALL gds.graph.project($name, 'Chunk', "
                f"{{{rel_type}: {{orientation: 'UNDIRECTED'}}}}) "
                "YIELD nodeCount, relationshipCount",
                name=graph_name,
            ).single()
        return (row["nodeCount"], row["relationshipCount"]) if row else (0, 0)

    def detect_communities(
        self, algorithm: str = "leiden", graph_name: str = "chunks", edge_type: str = "lexical",
    ) -> dict:
        """Runs community detection and returns per-community sizes + the
        doc_ids inside each (for the caller to label/color by source — this
        adapter has no knowledge of Qdrant/source_code, see
        services/api_gateway/routers/corpus.py for that join), plus
        inter-community edge weights for the community-level graph view.
        """
        if algorithm not in self._COMMUNITY_ALGORITHMS:
            raise ValueError(
                f"unsupported algorithm {algorithm!r}, expected one of {self._COMMUNITY_ALGORITHMS}"
            )
        if edge_type not in self._COMMUNITY_EDGE_TYPES:
            raise ValueError(
                f"unsupported edge_type {edge_type!r}, expected one of {tuple(self._COMMUNITY_EDGE_TYPES)}"
            )
        # algorithm/edge_type are checked against allowlists above before use
        # in the f-strings below — never raw user input reaching Cypher (same
        # pattern as max_keyword_freq in link_related()).
        rel_type = self._COMMUNITY_EDGE_TYPES[edge_type]
        # Suffixed by edge_type so running both signals back-to-back doesn't
        # have one overwrite the other's writeProperty — the whole point is
        # comparing them, not losing one (see link_semantic's docstring).
        write_property = f"_{algorithm}Community_{edge_type}"
        n_nodes, n_rels = self.project_graph(graph_name, edge_type=edge_type)
        # GDS fails from the inside instead of returning an empty result: on a
        # projection with no edges `gds.leiden.write` throws a
        # NullPointerException ("LeidenResult.dendrogramManager() is null"),
        # which surfaces as a 500 with no explanation. Asking the projection for
        # its size costs less than reading somebody else's stack trace, and the
        # answer here is not an error: a graph with no relationships has no
        # communities.
        if n_rels == 0:
            # The same shape as the successful response below. A reader of
            # this dict asks for `algorithm` and `edge_type` without checking
            # whether any community was found, so an empty response missing
            # them would be a second shape every caller has to know about
            # separately.
            return {
                "algorithm": algorithm,
                "edge_type": edge_type,
                "community_count": 0,
                "modularity": None,
                "communities": [],
                "inter_community_edges": [],
                "node_count": n_nodes,
                "relationship_count": 0,
            }
        driver = self._connect()
        with driver.session() as session:
            stats = session.run(
                f"CALL gds.{algorithm}.write($name, {{writeProperty: $prop}}) "
                "YIELD communityCount, modularity",
                name=graph_name, prop=write_property,
            ).single()

            community_rows = session.run(
                f"MATCH (n:Chunk) "
                f"RETURN n.{write_property} AS community_id, count(*) AS size, "
                "collect(DISTINCT n.doc_id) AS doc_ids "
                "ORDER BY size DESC"
            )
            communities = [
                {"community_id": r["community_id"], "size": r["size"], "doc_ids": r["doc_ids"]}
                for r in community_rows
            ]

            edge_rows = session.run(
                f"MATCH (a:Chunk)-[:{rel_type}]-(b:Chunk) "
                f"WHERE a.{write_property} < b.{write_property} "
                f"RETURN a.{write_property} AS community_a, b.{write_property} AS community_b, "
                "count(*) AS weight"
            )
            inter_community_edges = [dict(r) for r in edge_rows]

        return {
            "algorithm": algorithm,
            "edge_type": edge_type,
            "community_count": stats["communityCount"] if stats else len(communities),
            "modularity": stats["modularity"] if stats else None,
            "communities": communities,
            "inter_community_edges": inter_community_edges,
        }

    def community_subgraph(
        self, community_id: int, algorithm: str = "leiden", edge_type: str = "lexical",
    ) -> dict:
        """Nodes + edges (of the given edge_type) within a single community,
        for drill-down.

        Assumes detect_communities(algorithm=..., edge_type=...) already ran
        and wrote the community property this call reads — a fresh
        detect_communities() call before drill-down keeps the two consistent
        across a UI session.
        """
        if algorithm not in self._COMMUNITY_ALGORITHMS:
            raise ValueError(
                f"unsupported algorithm {algorithm!r}, expected one of {self._COMMUNITY_ALGORITHMS}"
            )
        if edge_type not in self._COMMUNITY_EDGE_TYPES:
            raise ValueError(
                f"unsupported edge_type {edge_type!r}, expected one of {tuple(self._COMMUNITY_EDGE_TYPES)}"
            )
        rel_type = self._COMMUNITY_EDGE_TYPES[edge_type]
        write_property = f"_{algorithm}Community_{edge_type}"
        driver = self._connect()
        with driver.session() as session:
            node_rows = session.run(
                f"MATCH (n:Chunk) WHERE n.{write_property} = $cid "
                "RETURN n.chunk_id AS chunk_id, n.doc_id AS doc_id, n.path AS path",
                cid=community_id,
            )
            nodes = [dict(r) for r in node_rows]
            edge_rows = session.run(
                f"MATCH (a:Chunk)-[:{rel_type}]-(b:Chunk) "
                f"WHERE a.{write_property} = $cid AND b.{write_property} = $cid AND a.chunk_id < b.chunk_id "
                "RETURN a.chunk_id AS source, b.chunk_id AS target",
                cid=community_id,
            )
            edges = [dict(r) for r in edge_rows]
        return {"community_id": community_id, "nodes": nodes, "edges": edges}

    def build_graph(self, chunks: list[Chunk]) -> int:
        """Full single-shot build: nodes + NEXT + RELATED for ``chunks``.

        Kept for backward compatibility and single-batch callers. For the
        ingest loop prefer ``add_chunks`` per file followed by one
        ``link_related`` — calling this per file re-links the entire graph and
        is O(files × n²).
        """
        self.ensure_indexes()
        n = self.add_chunks(chunks)
        self.link_related()
        return n

    # ── retrieval ──────────────────────────────────────────────────────────────

    def retrieve_graph(
        self,
        query: str,
        k: int = 5,
        hops: int = 1,
        filters: dict[str, Any] | None = None,
    ) -> GraphResult:
        driver = self._connect()
        terms = _keywords(query)
        # Neo4j forbids a parameter inside a variable-length bound `[*1..$hops]`; it
        # must be a literal. hops/k are ours (from config), but cast to int defensively.
        hops_lit = max(1, min(int(hops), 5))
        k_lit = max(1, min(int(k), 100))
        cypher = (
            "MATCH (seed:Chunk) "
            "WHERE any(k IN seed.keywords WHERE k IN $terms) "
            f"WITH seed LIMIT {k_lit} "
            f"OPTIONAL MATCH p=(seed)-[*1..{hops_lit}]-(nb:Chunk) "
            "WITH collect(DISTINCT seed) AS seeds, collect(DISTINCT nb) AS nbs "
            "RETURN seeds, nbs"
        )
        nodes: dict[str, GraphNode] = {}
        chunks: list[ScoredChunk] = []
        with driver.session() as session:
            rec = session.run(cypher, terms=terms).single()
            seeds = (rec["seeds"] if rec else []) or []
            nbs = (rec["nbs"] if rec else []) or []
            for rank, node in enumerate(seeds[:k]):
                cid = node["chunk_id"]
                score = 1.0 - rank / max(k, 1)
                chunks.append(
                    ScoredChunk(
                        chunk=Chunk(
                            doc_id=node.get("doc_id", cid),
                            text=node.get("text", ""),
                            chunk_id=cid,
                            structural_path=node.get("path", ""),
                            strategy_id="graph",
                        ),
                        score=score,
                        retriever_id=self.retriever_id,
                    )
                )
            for node in list(seeds) + list(nbs):
                cid = node["chunk_id"]
                nodes.setdefault(cid, GraphNode(node_id=cid, label=node.get("path", cid)))
        return GraphResult(nodes=list(nodes.values()), edges=[], scored_chunks=chunks)
