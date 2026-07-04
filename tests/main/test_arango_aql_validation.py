"""Unit tests for the AQL safety validator used by ArangoChatAgent.

These cover the ArangoDB arm of GHSA-2pq5-3q89-j7cc: LLM-generated AQL must not
be able to modify or destroy data via the read (retrieval) path, nor invoke
user-defined functions (server-side JavaScript), unless the operator opts in
with ``allow_dangerous_operations=True``. Pure-function tests, no live ArangoDB
required.
"""

import pytest

from langroid.agent.special.arangodb.aql_validator import (
    _strip_literals_and_comments,
    validate_aql_query,
)

# Read-only AQL that must pass on the retrieval path.
READ_OK = [
    "FOR doc IN users RETURN doc",
    "FOR v, e, p IN 1..1 OUTBOUND 'users/Bob' GRAPH 'family_tree' RETURN v",
    "FOR doc IN Actor LIMIT 1 RETURN ATTRIBUTES(doc)",
    "FOR d IN users FILTER d.updated > 0 RETURN d",  # `.updated`, not UPDATE
    # write keywords appearing only inside strings / comments:
    "FOR d IN users FILTER d.note == 'INSERT this' RETURN d",
    "FOR d IN users RETURN d // UPDATE later",
    "FOR d IN users RETURN d /* no REMOVE here */",
]

# Queries that must be rejected on the read path (writes + UDF calls).
READ_REJECT = [
    "INSERT { name: 'x' } INTO users",
    "FOR u IN users UPDATE u WITH { age: 1 } IN users",
    "FOR u IN users REPLACE u WITH { name: 'x' } IN users",
    "FOR u IN users REMOVE u IN users",
    "UPSERT { name: 'x' } INSERT { name: 'x' } UPDATE { age: 1 } IN users",
    "FOR x IN 1..1 RETURN MYGROUP::EVIL(x)",
]

# Ordinary writes that the creation path should allow.
WRITE_OK = [
    "INSERT { name: 'x' } INTO users",
    "FOR u IN users REMOVE u IN users",
    "FOR u IN users UPDATE u WITH { age: 31 } IN users",
    "UPSERT { name: 'x' } INSERT { name: 'x' } UPDATE { age: 1 } IN users",
]

# UDF calls that must be blocked even on the creation path.
WRITE_REJECT = [
    "FOR x IN 1..1 RETURN MYGROUP::EVIL(x)",
    "INSERT { v: MYLIB::run('whoami') } INTO users",
]


@pytest.mark.parametrize("query", READ_OK)
def test_read_path_allows_read_only(query: str) -> None:
    assert validate_aql_query(query, is_write=False, allow_dangerous=False) is None


@pytest.mark.parametrize("query", READ_REJECT)
def test_read_path_rejects_writes_and_dangerous(query: str) -> None:
    assert validate_aql_query(query, is_write=False, allow_dangerous=False) is not None


@pytest.mark.parametrize("query", WRITE_OK)
def test_write_path_allows_writes(query: str) -> None:
    assert validate_aql_query(query, is_write=True, allow_dangerous=False) is None


@pytest.mark.parametrize("query", WRITE_REJECT)
def test_write_path_rejects_dangerous(query: str) -> None:
    assert validate_aql_query(query, is_write=True, allow_dangerous=False) is not None


@pytest.mark.parametrize("query", READ_REJECT + WRITE_REJECT)
def test_allow_dangerous_bypasses_all_checks(query: str) -> None:
    assert validate_aql_query(query, is_write=False, allow_dangerous=True) is None
    assert validate_aql_query(query, is_write=True, allow_dangerous=True) is None


def test_strip_literals_and_comments() -> None:
    # keyword inside a string literal is blanked
    assert "INSERT" not in _strip_literals_and_comments(
        "FOR d IN users FILTER d.x == 'INSERT' RETURN d"
    )
    # keyword inside a line comment is blanked
    assert "REMOVE" not in _strip_literals_and_comments(
        "FOR d IN users RETURN d // REMOVE"
    )
    # keyword inside a block comment is blanked
    assert "UPDATE" not in _strip_literals_and_comments("RETURN 1 /* UPDATE x */")
    # real clause text is preserved
    assert "RETURN" in _strip_literals_and_comments("FOR d IN users RETURN d")
