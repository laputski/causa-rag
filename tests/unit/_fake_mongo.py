"""A Mongo stand-in with the few operators the run store uses.

Real matching semantics where they matter: an absent field never equals the
value asked for, and `$in`, `$lt` and `$ne` behave as the engine's do on plain
values. Anything else it is asked raises, so a query this does not understand
fails the test instead of matching everything.
"""
from __future__ import annotations

import copy
import json
from typing import Any


class FakeMongo:
    def __init__(self, data: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self.data: dict[str, list[dict[str, Any]]] = {
            k: [copy.deepcopy(d) for d in v] for k, v in (data or {}).items()}

    # ── matching ──────────────────────────────────────────────────────────────

    def _field(self, doc: dict[str, Any], key: str) -> tuple[bool, Any]:
        cur: Any = doc
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return False, None
            cur = cur[part]
        return True, cur

    def _match(self, doc: dict[str, Any], query: dict[str, Any]) -> bool:
        for key, want in query.items():
            present, value = self._field(doc, key)
            if isinstance(want, dict) and want and all(k.startswith("$") for k in want):
                for op, arg in want.items():
                    if op == "$in":
                        if not present or value not in arg:
                            return False
                    elif op == "$nin":
                        if present and value in arg:
                            return False
                    elif op == "$lt":
                        if not present or not value < arg:
                            return False
                    elif op == "$ne":
                        if present and value == arg:
                            return False
                    elif op == "$exists":
                        if present != bool(arg):
                            return False
                    else:
                        raise NotImplementedError(op)
            elif not present or value != want:
                return False
        return True

    def _docs(self, collection: str) -> list[dict[str, Any]]:
        return self.data.setdefault(collection, [])

    def _read(self, collection: str) -> list[dict[str, Any]]:
        # A read never creates the collection it looks in.
        return self.data.get(collection, [])

    # ── the adapter's surface ─────────────────────────────────────────────────

    async def find_one(self, collection: str, query: dict[str, Any] | None = None,
                       projection: dict[str, Any] | None = None) -> dict[str, Any] | None:
        for d in self._read(collection):
            if self._match(d, query or {}):
                return self._project(d, projection)
        return None

    async def find_many(self, collection: str, query: dict[str, Any] | None = None,
                        sort: list[tuple[str, int]] | None = None, limit: int = 0,
                        projection: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        docs = [d for d in self._read(collection) if self._match(d, query or {})]
        for key, direction in reversed(sort or []):
            docs.sort(key=lambda d, k=key: str(d.get(k, "")), reverse=direction < 0)
        if limit:
            docs = docs[:limit]
        return [self._project(d, projection) for d in docs]

    def _project(self, doc: dict[str, Any], projection: dict[str, Any] | None) -> dict[str, Any]:
        out = copy.deepcopy(doc)
        if not projection:
            return out
        if all(v == 0 for v in projection.values()):
            return {k: v for k, v in out.items() if k not in projection}
        kept: dict[str, Any] = {"_id": out["_id"]} if "_id" in out else {}
        for path in projection:
            self._keep(out, kept, path.split("."))
        return kept

    def _keep(self, source: Any, target: dict[str, Any], parts: list[str]) -> None:
        """An inclusion path the way the engine applies it, through arrays."""
        head, rest = parts[0], parts[1:]
        if not isinstance(source, dict) or head not in source:
            return
        value = source[head]
        if not rest:
            target[head] = value
        elif isinstance(value, list):
            items = target.setdefault(head, [{} for _ in value])
            for item, into in zip(value, items, strict=True):
                self._keep(item, into, rest)
        elif isinstance(value, dict):
            self._keep(value, target.setdefault(head, {}), rest)

    async def insert_one(self, collection: str, doc: dict[str, Any]) -> str:
        self._docs(collection).append(copy.deepcopy(doc))
        return str(len(self._docs(collection)))

    async def upsert_one(self, collection: str, query: dict[str, Any], doc: dict[str, Any]) -> None:
        docs = self._docs(collection)
        docs[:] = [d for d in docs if not self._match(d, query)]
        docs.append(copy.deepcopy(doc))

    async def update_one(self, collection: str, query: dict[str, Any],
                         update: dict[str, Any], upsert: bool = False) -> int:
        for d in self._docs(collection):
            if self._match(d, query):
                d.update(copy.deepcopy(update.get("$set", {})))
                return 1
        if upsert:
            self._docs(collection).append({**query, **copy.deepcopy(update.get("$set", {}))})
        return 0

    async def bulk_upsert(self, collection: str,
                          ops: list[tuple[dict[str, Any], dict[str, Any]]]) -> None:
        for query, update in ops:
            existing = next((d for d in self._docs(collection) if self._match(d, query)), None)
            if existing is None:
                existing = dict(query)
                existing.update(copy.deepcopy(update.get("$setOnInsert", {})))
                self._docs(collection).append(existing)
            existing.update(copy.deepcopy(update.get("$set", {})))

    async def delete_one(self, collection: str, query: dict[str, Any]) -> int:
        docs = self._docs(collection)
        for i, d in enumerate(docs):
            if self._match(d, query):
                del docs[i]
                return 1
        return 0

    async def delete_many(self, collection: str, query: dict[str, Any]) -> int:
        if collection not in self.data:
            return 0
        docs = self._docs(collection)
        kept = [d for d in docs if not self._match(d, query)]
        removed = len(docs) - len(kept)
        docs[:] = kept
        return removed

    async def count(self, collection: str, query: dict[str, Any] | None = None) -> int:
        return sum(1 for d in self._read(collection) if self._match(d, query or {}))

    async def data_size(self, collection: str) -> int:
        return sum(self.doc_size(d) for d in self._read(collection))

    async def sizes_by(self, collection: str, field: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for d in self._read(collection):
            key = str(d.get(field, ""))
            out[key] = out.get(key, 0) + self.doc_size(d)
        return out

    @staticmethod
    def doc_size(doc: dict[str, Any]) -> int:
        return len(json.dumps(doc, ensure_ascii=False, default=str).encode())

    def install(self, monkeypatch: Any) -> FakeMongo:
        import adapters.mongodb as mdb
        for name in ("find_one", "find_many", "insert_one", "upsert_one", "update_one",
                     "bulk_upsert", "delete_one", "delete_many", "count",
                     "data_size", "sizes_by"):
            monkeypatch.setattr(mdb, name, getattr(self, name), raising=False)
        return self
