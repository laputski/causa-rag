"""MongoDB adapter — async Motor client for RAG platform persistence."""
from __future__ import annotations

import os
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

_MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
_DB_NAME = os.getenv("MONGODB_DB", "ragplatform")

_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(_MONGODB_URL)
    return _client


def get_collection(name: str) -> AsyncIOMotorCollection:
    return get_client()[_DB_NAME][name]


def _match_id(query: dict[str, Any]) -> dict[str, Any]:
    """Allow a lookup by the same `_id` this module handed out.

    `find_one`/`find_many` convert `_id` to a string, because otherwise it does
    not survive JSON. Nothing converted it back, so a query of
    `{"_id": "6a88…"}` matched no document whose real `_id` is an ObjectId with
    those same characters. Silently: `delete_one` returned zero and the caller
    counted the document as deleted.

    That is how the end-to-end test's cleanups went missing, and why seven
    realms had accumulated. The match is checked against both representations at
    once, because some collections (`settings`) really do store a string in
    `_id` and must not be broken.
    """
    raw = query.get("_id")
    if not isinstance(raw, str) or not ObjectId.is_valid(raw):
        return query
    return {**query, "_id": {"$in": [ObjectId(raw), raw]}}


async def insert_one(collection: str, doc: dict[str, Any]) -> str:
    result = await get_collection(collection).insert_one(doc)
    return str(result.inserted_id)


async def find_one(
    collection: str,
    query: dict[str, Any],
    projection: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    doc = await get_collection(collection).find_one(_match_id(query), projection)
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


async def find_many(
    collection: str,
    query: dict[str, Any] | None = None,
    sort: list[tuple[str, int]] | None = None,
    limit: int = 0,
    projection: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Documents matching the query.

    `projection` names which fields come back, in the engine's own form: a
    map of field to 1 to take only those, or to 0 to leave those behind. It is
    what makes "every run without its question results" a cheap read rather
    than the whole store parsed to reach five fields of each document. Absent,
    and every field arrives, which is what every caller before it got.
    """
    cursor = get_collection(collection).find(_match_id(query or {}), projection)
    if sort:
        cursor = cursor.sort(sort)
    if limit:
        cursor = cursor.limit(limit)
    docs = await cursor.to_list(length=limit or 1000)
    for doc in docs:
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])
    return docs


async def update_one(
    collection: str, query: dict[str, Any], update: dict[str, Any], upsert: bool = False
) -> int:
    result = await get_collection(collection).update_one(_match_id(query), update, upsert=upsert)
    return result.modified_count


async def upsert_one(collection: str, query: dict[str, Any], doc: dict[str, Any]) -> None:
    await get_collection(collection).replace_one(query, doc, upsert=True)


async def delete_one(collection: str, query: dict[str, Any]) -> int:
    result = await get_collection(collection).delete_one(_match_id(query))
    return result.deleted_count


async def delete_many(collection: str, query: dict[str, Any]) -> int:
    result = await get_collection(collection).delete_many(_match_id(query))
    return result.deleted_count


async def count(collection: str, query: dict[str, Any] | None = None) -> int:
    return await get_collection(collection).count_documents(query or {})
