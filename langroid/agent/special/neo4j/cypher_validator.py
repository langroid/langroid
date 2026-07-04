"""Security validation for LLM-generated Cypher queries (GHSA-2pq5-3q89-j7cc).

``Neo4jChatAgent`` executes Cypher produced by an LLM, whose text is
influenceable by prompt injection -- directly, or indirectly via content the
agent reads back through RAG. Executing such Cypher without restriction lets an
attacker who can influence the prompt read or destroy all graph data and, when
APOC or ``dbms.security`` procedures are enabled on the database role, reach the
filesystem, network, or OS (config-conditional RCE). This mirrors the
prompt-to-SQL-to-RCE issue fixed for ``SQLChatAgent`` in CVE-2026-25879.

Unless the operator opts in with ``allow_dangerous_operations=True``:

- the retrieval (read) path is restricted to read-only Cypher -- any
  write or schema-mutating clause is rejected;
- both paths reject the code-execution / file / network primitives
  (``LOAD CSV``, ``apoc.*``, ``dbms.*``, ``CALL db.*``).

Cypher has no lightweight AST parser in our dependency set, so the checks are
keyword/pattern based, applied after stripping comments and string/identifier
literals to reduce false positives. A blocklist is inherently bypassable; the
real guarantee is the default-off gate plus running the agent against a
least-privilege Neo4j role. Treat this as defense in depth, not a parser-grade
allowlist.
"""

import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# How a rejected query can be allowed by the operator. Appended to every
# rejection message relayed back to the LLM.
_CONFIG_HINT = (
    "ask the operator to set `allow_dangerous_operations=True` on the "
    "Neo4jChatAgentConfig (only safe with a least-privilege Neo4j role and "
    "trusted prompts)"
)

# Primitives that enable code execution, filesystem/network access, or admin
# control regardless of read vs write context.
_DANGEROUS_PATTERNS: List[Tuple["re.Pattern[str]", str]] = [
    (re.compile(r"\bLOAD\s+CSV\b", re.IGNORECASE), "LOAD CSV file/URL access"),
    (re.compile(r"\bapoc\.", re.IGNORECASE), "an apoc.* procedure/function"),
    (re.compile(r"\bdbms\.", re.IGNORECASE), "a dbms.* admin procedure"),
    (re.compile(r"\bCALL\s+db\.", re.IGNORECASE), "a CALL db.* admin procedure"),
]

# Clauses that modify the graph; rejected only on the read (retrieval) path.
# ``(?<!\.)`` avoids matching property access such as ``n.set`` / ``n.create``
# (a reserved word used as a property key must be back-ticked, which the
# literal stripper blanks out).
_WRITE_PATTERNS: List[Tuple["re.Pattern[str]", str]] = [
    (re.compile(r"(?<!\.)\bCREATE\b", re.IGNORECASE), "CREATE"),
    (re.compile(r"(?<!\.)\bMERGE\b", re.IGNORECASE), "MERGE"),
    (re.compile(r"(?<!\.)\bSET\b", re.IGNORECASE), "SET"),
    (re.compile(r"(?<!\.)\bDELETE\b", re.IGNORECASE), "DELETE"),
    (re.compile(r"(?<!\.)\bREMOVE\b", re.IGNORECASE), "REMOVE"),
    (re.compile(r"(?<!\.)\bDROP\b", re.IGNORECASE), "DROP"),
    (re.compile(r"(?<!\.)\bFOREACH\b", re.IGNORECASE), "FOREACH"),
]


def _strip_literals_and_comments(query: str) -> str:
    """Blank out comments and string/backtick literals in ``query``.

    Keyword and pattern checks run against the result so they never match text
    inside string literals, back-ticked identifiers, ``//`` line comments, or
    ``/* ... */`` block comments. Replaced spans become spaces of equal length
    so character offsets are preserved. String literals use backslash escaping;
    back-ticked identifiers escape a backtick by doubling it.

    Args:
        query: Raw Cypher text.

    Returns:
        The query with comment and literal spans replaced by spaces.
    """
    out: List[str] = []
    i, n = 0, len(query)
    while i < n:
        two = query[i : i + 2]
        if two == "//":
            j = query.find("\n", i)
            j = n if j == -1 else j
            out.append(" " * (j - i))
            i = j
        elif two == "/*":
            j = query.find("*/", i + 2)
            j = n if j == -1 else j + 2
            out.append(" " * (j - i))
            i = j
        elif query[i] in "'\"":
            ch = query[i]
            j = i + 1
            while j < n:
                if query[j] == "\\":
                    j += 2
                    continue
                if query[j] == ch:
                    break
                j += 1
            j = min(j + 1, n)
            out.append(" " * (j - i))
            i = j
        elif query[i] == "`":
            j = i + 1
            while j < n:
                if query[j] == "`":
                    if j + 1 < n and query[j + 1] == "`":
                        j += 2
                        continue
                    break
                j += 1
            j = min(j + 1, n)
            out.append(" " * (j - i))
            i = j
        else:
            out.append(query[i])
            i += 1
    return "".join(out)


def validate_cypher_query(
    query: str, *, is_write: bool, allow_dangerous: bool
) -> Optional[str]:
    """Check an LLM-generated Cypher query against the agent's safety policy.

    Args:
        query: The raw Cypher string from the LLM.
        is_write: True for the creation/write path (ordinary write clauses are
            permitted), False for the retrieval/read path (write clauses are
            rejected so the path is read-only).
        allow_dangerous: When True, skip all checks (operator opt-in).

    Returns:
        None if the query may be executed, otherwise a human-readable rejection
        message to relay back to the LLM.
    """
    if allow_dangerous:
        return None

    scrubbed = _strip_literals_and_comments(query)

    for pat, label in _DANGEROUS_PATTERNS:
        if pat.search(scrubbed):
            logger.warning("Neo4jChatAgent rejected Cypher using %s: %r", label, query)
            return (
                f"Cypher query REJECTED for safety: it uses {label}, which can "
                f"execute code or access the filesystem/network. Rewrite the "
                f"query without it, or {_CONFIG_HINT}."
            )

    if not is_write:
        for pat, label in _WRITE_PATTERNS:
            if pat.search(scrubbed):
                logger.warning(
                    "Neo4jChatAgent rejected write clause %s on read path: %r",
                    label,
                    query,
                )
                return (
                    f"Cypher query REJECTED for safety: the retrieval tool runs "
                    f"READ-ONLY Cypher, but this query uses `{label}`. Use the "
                    f"creation tool for writes, rewrite this as a read-only "
                    f"(MATCH/RETURN) query, or {_CONFIG_HINT}."
                )

    return None
