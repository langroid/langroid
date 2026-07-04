"""Security validation for LLM-generated AQL queries (GHSA-2pq5-3q89-j7cc).

``ArangoChatAgent`` executes AQL produced by an LLM, whose text is influenceable
by prompt injection -- directly, or indirectly via content the agent reads back
through RAG. ``bind_vars`` only parameterizes *values*, so the statement text is
the raw LLM string: unrestricted, an attacker who can influence the prompt can
read or destroy all graph data and, where user-defined AQL functions (UDFs),
Foxx, or ``javascript.allow-admin-execute`` are enabled on the database role,
escalate toward RCE. This mirrors the prompt-to-SQL-to-RCE issue fixed for
``SQLChatAgent`` in CVE-2026-25879 and the sibling Cypher issue in
``Neo4jChatAgent``.

Unless the operator opts in with ``allow_dangerous_operations=True``:

- the retrieval (read) path is restricted to read-only AQL -- any
  data-modification operation (INSERT/UPDATE/REPLACE/REMOVE/UPSERT) is rejected;
- both paths reject user-defined-function calls (``namespace::func``), which can
  run server-side JavaScript.

The checks are keyword/pattern based, applied after stripping comments and
string/identifier literals to reduce false positives. A blocklist is inherently
bypassable; the real guarantee is the default-off gate plus running the agent
against a least-privilege ArangoDB role. Treat this as defense in depth, not a
parser-grade allowlist.
"""

import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# How a rejected query can be allowed by the operator. Appended to every
# rejection message relayed back to the LLM.
_CONFIG_HINT = (
    "ask the operator to set `allow_dangerous_operations=True` on the "
    "ArangoChatAgentConfig (only safe with a least-privilege ArangoDB role and "
    "trusted prompts)"
)

# Primitives that can run server-side code, regardless of read vs write context.
# AQL user-defined functions are called as ``namespace::function(...)`` and can
# execute registered JavaScript.
_DANGEROUS_PATTERNS: List[Tuple["re.Pattern[str]", str]] = [
    (
        re.compile(r"[A-Za-z0-9_]+\s*::\s*[A-Za-z0-9_]+"),
        "a user-defined function call (namespace::func)",
    ),
]

# Data-modification operations; rejected only on the read (retrieval) path.
# ``(?<!\.)`` avoids matching attribute access such as ``doc.update``.
_WRITE_PATTERNS: List[Tuple["re.Pattern[str]", str]] = [
    (re.compile(r"(?<!\.)\bINSERT\b", re.IGNORECASE), "INSERT"),
    (re.compile(r"(?<!\.)\bUPDATE\b", re.IGNORECASE), "UPDATE"),
    (re.compile(r"(?<!\.)\bREPLACE\b", re.IGNORECASE), "REPLACE"),
    (re.compile(r"(?<!\.)\bREMOVE\b", re.IGNORECASE), "REMOVE"),
    (re.compile(r"(?<!\.)\bUPSERT\b", re.IGNORECASE), "UPSERT"),
]


def _strip_literals_and_comments(query: str) -> str:
    """Blank out comments and string/identifier literals in ``query``.

    Keyword and pattern checks run against the result so they never match text
    inside string literals, back-tick / forward-tick identifiers, ``//`` line
    comments, or ``/* ... */`` block comments. Replaced spans become spaces of
    equal length so character offsets are preserved. String literals use
    backslash escaping.

    Args:
        query: Raw AQL text.

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
        elif query[i] in "`´":
            ch = query[i]
            j = query.find(ch, i + 1)
            j = n if j == -1 else j + 1
            out.append(" " * (j - i))
            i = j
        else:
            out.append(query[i])
            i += 1
    return "".join(out)


def validate_aql_query(
    query: str, *, is_write: bool, allow_dangerous: bool
) -> Optional[str]:
    """Check an LLM-generated AQL query against the agent's safety policy.

    Args:
        query: The raw AQL string from the LLM.
        is_write: True for the creation/write path (data-modification
            operations are permitted), False for the retrieval/read path
            (modifications are rejected so the path is read-only).
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
            logger.warning("ArangoChatAgent rejected AQL using %s: %r", label, query)
            return (
                f"AQL query REJECTED for safety: it uses {label}, which can "
                f"execute server-side code. Rewrite the query without it, or "
                f"{_CONFIG_HINT}."
            )

    if not is_write:
        for pat, label in _WRITE_PATTERNS:
            if pat.search(scrubbed):
                logger.warning(
                    "ArangoChatAgent rejected write op %s on read path: %r",
                    label,
                    query,
                )
                return (
                    f"AQL query REJECTED for safety: the retrieval tool runs "
                    f"READ-ONLY AQL, but this query uses `{label}`. Use the "
                    f"creation tool for writes, rewrite this as a read-only "
                    f"(FOR ... RETURN) query, or {_CONFIG_HINT}."
                )

    return None
