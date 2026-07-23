"""Tests that ``CSVGraphAgent.pandas_to_kg`` applies the Cypher validation gate.

Regression tests for GHSA-83w4-crcp-3w4p / GHSA-mv6j-6w86-7ph9: the
``pandas_to_kg`` handler called ``write_query`` directly on LLM-generated
Cypher, skipping the ``validate_cypher_query`` gate that
``Neo4jChatAgent.cypher_creation_tool`` applies.

The handler is exercised against a minimal stand-in holding only the two
attributes it reads (``config`` and ``df``), so no live Neo4j instance is
needed; ``validate_cypher_query`` itself is the real implementation.
"""

from typing import Any, Dict, List, Optional

import pytest

from langroid.agent.special.neo4j.csv_kg_chat import CSVGraphAgent, PandasToKGTool


class _Config:
    def __init__(self, allow_dangerous: bool) -> None:
        self.allow_dangerous_operations = allow_dangerous


class _Response:
    success = True
    data: List[Any] = []


class _StubAgent:
    """Minimal stand-in exposing what ``pandas_to_kg`` actually touches."""

    def __init__(self, allow_dangerous: bool = False) -> None:
        self.config = _Config(allow_dangerous)
        self.df = None  # no rows => no execution beyond the gate
        self.executed: List[str] = []

    def write_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> _Response:
        self.executed.append(query)
        return _Response()


DANGEROUS = [
    "CALL apoc.load.json('file:///etc/passwd') YIELD value RETURN value",
    "CALL dbms.security.listUsers()",
    "LOAD CSV FROM 'file:///etc/passwd' AS line RETURN line",
]


@pytest.mark.parametrize("query", DANGEROUS)
def test_pandas_to_kg_rejects_dangerous_cypher(query: str) -> None:
    agent = _StubAgent()
    msg = PandasToKGTool(cypherQuery=query, args=[])
    result = CSVGraphAgent.pandas_to_kg(agent, msg)  # type: ignore[arg-type]

    assert "REJECTED" in result.upper()
    assert agent.executed == [], "dangerous Cypher reached write_query"


def test_pandas_to_kg_allows_benign_cypher() -> None:
    agent = _StubAgent()
    msg = PandasToKGTool(
        cypherQuery="MERGE (p:Person {name: $name})",
        args=["name"],
    )
    result = CSVGraphAgent.pandas_to_kg(agent, msg)  # type: ignore[arg-type]

    assert "REJECTED" not in result.upper()


def test_allow_dangerous_operations_bypasses_the_gate() -> None:
    """The documented opt-in still disables the gate, as elsewhere."""
    agent = _StubAgent(allow_dangerous=True)
    msg = PandasToKGTool(cypherQuery=DANGEROUS[0], args=[])
    result = CSVGraphAgent.pandas_to_kg(agent, msg)  # type: ignore[arg-type]

    assert "REJECTED" not in result.upper()
