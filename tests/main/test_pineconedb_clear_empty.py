"""Tests for `PineconeDB.clear_empty_collections`.

These exercise the real method bodies against a stub Pinecone client, so they
need neither the `pinecone` package nor a `PINECONE_API_KEY`, and make no
network or LLM calls.
"""

from typing import Dict, List

import pytest

import langroid.vector_store.pineconedb as pineconedb
from langroid.vector_store.pineconedb import PineconeDB

# `_fetch_index_meta` reports this count for an index whose stats it could not
# read, having caught a `PineconeApiException`.
STATS_UNREADABLE = -1


class _StubIndex:
    """Stands in for `pinecone.Index`, serving a fixed vector count."""

    def __init__(self, total_vector_count: int) -> None:
        self.total_vector_count = total_vector_count

    def describe_index_stats(self) -> Dict[str, int]:
        return {"total_vector_count": self.total_vector_count}


class _StubIndexList:
    """Stands in for the object returned by `Pinecone.list_indexes`."""

    def __init__(self, names: List[str]) -> None:
        self._names = names

    def names(self) -> List[str]:
        return self._names


class _StubPineconeClient:
    """Minimal stand-in for `pinecone.Pinecone`, recording deletions.

    Args:
        counts: Maps index name to the vector count its stats report.
    """

    def __init__(self, counts: Dict[str, int]) -> None:
        self.counts = counts
        self.deleted: List[str] = []

    def list_indexes(self) -> _StubIndexList:
        return _StubIndexList(list(self.counts))

    def Index(self, name: str) -> _StubIndex:
        return _StubIndex(self.counts[name])

    def delete_index(self, name: str) -> None:
        self.deleted.append(name)


def _db_with_client(counts: Dict[str, int]) -> PineconeDB:
    """Build a `PineconeDB` wired to a stub client, skipping `__init__`.

    `PineconeDB.__init__` requires the `pinecone` package and an API key; the
    method under test only touches `self.client`.
    """
    db = PineconeDB.__new__(PineconeDB)
    db.client = _StubPineconeClient(counts)  # type: ignore[assignment]
    return db


def test_clear_empty_collections_keeps_non_empty_indexes() -> None:
    """Only indexes with zero vectors are deleted."""
    db = _db_with_client({"empty-idx": 0, "full-idx": 42, "other-empty": 0})

    n_deletes = db.clear_empty_collections()

    assert sorted(db.client.deleted) == ["empty-idx", "other-empty"]
    assert n_deletes == 2


def test_clear_empty_collections_keeps_index_with_unreadable_stats() -> None:
    """An index whose vector count could not be read is left alone.

    There is no evidence such an index is empty, so deleting it could discard
    data.
    """
    db = _db_with_client({"empty-idx": 0, "unreadable-idx": STATS_UNREADABLE})

    n_deletes = db.clear_empty_collections()

    assert db.client.deleted == ["empty-idx"]
    assert n_deletes == 1


def test_clear_empty_collections_no_empty_indexes_is_a_no_op() -> None:
    """With every index populated, nothing is deleted."""
    db = _db_with_client({"full-a": 1, "full-b": 7})

    assert db.clear_empty_collections() == 0
    assert db.client.deleted == []


def test_unreadable_stats_index_survives_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An index whose stats call raises is reported as `-1`, and kept.

    This drives the real `except PineconeApiException` path in
    `_fetch_index_meta`, which is what produces the `-1` the test above
    assumes. Without the `pinecone` package installed, the module binds
    `PineconeApiException` to `typing.Any`, so substitute a real exception
    class — what the name refers to when `pinecone` is present.
    """

    class _FakeApiException(Exception):
        pass

    monkeypatch.setattr(pineconedb, "PineconeApiException", _FakeApiException)

    class _RaisingClient(_StubPineconeClient):
        def Index(self, name: str) -> _StubIndex:
            if name == "unreadable-idx":
                raise _FakeApiException("stats unavailable")
            return super().Index(name)

    db = PineconeDB.__new__(PineconeDB)
    db.client = _RaisingClient({"empty-idx": 0, "unreadable-idx": 0})  # type: ignore[assignment]

    assert db._fetch_index_meta("unreadable-idx").total_vector_count == STATS_UNREADABLE
    assert db.clear_empty_collections() == 1
    assert db.client.deleted == ["empty-idx"]
