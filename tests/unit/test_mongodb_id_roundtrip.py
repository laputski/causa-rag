"""What the adapter hands out as an `_id`, the adapter must accept back.

`find_one` and `find_many` convert `_id` to a string, because otherwise it does
not survive JSON. Nobody converted it back, so `delete_one({"_id": doc["_id"]})`
matched nothing, and matched nothing silently: zero deleted, while the caller
believed the document was gone.

That is how end-to-end cleanups quietly stopped working. Seven realms
accumulated in the list for exactly this reason, and for the same reason the
script written to remove them did not remove them.
"""
from __future__ import annotations

from bson import ObjectId

from adapters.mongodb import _match_id


# @lat: [[data-backends#MongoDB — то, что адаптер выдаёт, он же и принимает#Идентификатор, выданный адаптером]]
def test_a_stringified_object_id_matches_the_real_one() -> None:
    oid = ObjectId()
    q = _match_id({"_id": str(oid)})
    assert q["_id"] == {"$in": [oid, str(oid)]}


# @lat: [[data-backends#MongoDB — то, что адаптер выдаёт, он же и принимает#Идентификатор, выданный адаптером]]
def test_a_real_string_id_still_matches_itself() -> None:
    # `settings` stores a realm id in `_id`, a genuine string. Converting it to
    # an ObjectId would break it, so the match is tried against both
    # representations rather than one chosen in advance.
    oid = ObjectId()
    assert _match_id({"_id": str(oid)})["_id"]["$in"][1] == str(oid)


# @lat: [[data-backends#MongoDB — то, что адаптер выдаёт, он же и принимает#Идентификатор, выданный адаптером]]
def test_an_id_that_is_not_an_object_id_is_left_exactly_as_it_is() -> None:
    assert _match_id({"_id": "acme"}) == {"_id": "acme"}
    assert _match_id({"realm_id": "acme"}) == {"realm_id": "acme"}
    assert _match_id({}) == {}


# @lat: [[data-backends#MongoDB — то, что адаптер выдаёт, он же и принимает#Идентификатор, выданный адаптером]]
def test_the_rest_of_the_query_survives() -> None:
    oid = ObjectId()
    q = _match_id({"_id": str(oid), "realm_id": "acme"})
    assert q["realm_id"] == "acme"


# @lat: [[data-backends#MongoDB — то, что адаптер выдаёт, он же и принимает#Идентификатор, выданный адаптером]]
def test_an_object_id_passed_as_itself_is_untouched() -> None:
    # A caller working with a real ObjectId loses nothing.
    oid = ObjectId()
    assert _match_id({"_id": oid}) == {"_id": oid}
