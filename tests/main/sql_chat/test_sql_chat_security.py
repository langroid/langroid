"""
Unit tests for SQLChatAgent's query validator (CVE-2026-25879 mitigation).

These tests exercise `_validate_query` and `run_query` directly without an
LLM, so they don't require API credentials.
"""

import pytest

from langroid.exceptions import LangroidImportError

try:
    from sqlalchemy import Column, Integer, String, create_engine
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
except ImportError as e:
    raise LangroidImportError(extra="sql", error=str(e))

import sqlglot
from sqlglot import expressions as sqlglot_exp

from langroid.agent.special.sql.sql_chat_agent import (
    SQLChatAgent,
    SQLChatAgentConfig,
    _nested_write_kinds,
)
from langroid.agent.special.sql.utils.tools import RunQueryTool

Base = declarative_base()


class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    s.add(Item(id=1, name="a"))
    s.commit()
    yield s
    s.close()


def _make_agent(session, **cfg_kwargs):
    cfg = SQLChatAgentConfig(
        database_session=session,
        llm=None,
        use_helper=False,
        **cfg_kwargs,
    )
    return SQLChatAgent(cfg)


# ---------------------------------------------------------------------------
# Validator unit tests
# ---------------------------------------------------------------------------


def test_select_allowed_by_default(session):
    agent = _make_agent(session)
    assert agent._validate_query("SELECT * FROM items") is None


@pytest.mark.parametrize(
    "query",
    [
        "DROP TABLE items",
        "CREATE TABLE x (id int)",
        "ALTER TABLE items ADD COLUMN y int",
        "UPDATE items SET name='b' WHERE id=1",
        "INSERT INTO items (id, name) VALUES (2, 'b')",
        "DELETE FROM items WHERE id=1",
        "TRUNCATE TABLE items",
    ],
)
def test_non_select_blocked_by_default(session, query):
    agent = _make_agent(session)
    rejection = agent._validate_query(query)
    assert rejection is not None
    assert "REJECTED" in rejection


def test_cve_2026_25879_poc_blocked(session):
    """The exact reproducer from the security advisory must be rejected."""
    agent = _make_agent(session)
    poc = (
        "DROP TABLE IF EXISTS log;\n"
        "CREATE TABLE log(content text);\n"
        "COPY log(content) FROM PROGRAM 'id';\n"
        "SELECT * FROM log;"
    )
    rejection = agent._validate_query(poc)
    assert rejection is not None
    assert "REJECTED" in rejection


@pytest.mark.parametrize(
    "query",
    [
        # PostgreSQL: command execution
        "COPY t FROM PROGRAM 'id'",
        "COPY t (c) FROM PROGRAM 'whoami'",
        # PostgreSQL: server-side file read
        "SELECT pg_read_server_files('/etc/passwd')",
        "SELECT pg_read_binary_file('/etc/shadow')",
        "SELECT pg_ls_dir('/')",
        "SELECT lo_import('/etc/passwd')",
        # MySQL: filesystem
        "SELECT * FROM items INTO OUTFILE '/tmp/x'",
        "SELECT * FROM items INTO DUMPFILE '/tmp/x'",
        "SELECT load_file('/etc/passwd')",
        "LOAD DATA INFILE '/etc/passwd' INTO TABLE items",
        # SQLite: arbitrary code / file access
        "SELECT load_extension('/tmp/evil.so')",
        "ATTACH DATABASE '/etc/passwd' AS p",
        # MSSQL: command execution
        "EXEC xp_cmdshell 'id'",
        "EXEC sp_OACreate 'WScript.Shell', @s OUT",
        "SELECT * FROM OPENROWSET('SQLNCLI', 'connstring', 'q')",
        "BULK INSERT t FROM '/etc/passwd'",
        # Generic: stored programs and extensions
        "CREATE FUNCTION evil() RETURNS void AS $$ ... $$ LANGUAGE plpgsql",
        "CREATE OR REPLACE PROCEDURE p() AS ...",
        "CREATE EXTENSION plpython3u",
    ],
)
def test_dangerous_patterns_blocked(session, query):
    agent = _make_agent(session)
    rejection = agent._validate_query(query)
    assert rejection is not None
    assert "REJECTED" in rejection


@pytest.mark.parametrize(
    "query",
    [
        # PostgreSQL: the pg_read_file / pg_stat_file / pg_ls_* /
        # pg_current_logfile family yields the same file/metadata disclosure
        # primitive as pg_read_server_file but uses different function names.
        "SELECT pg_read_file('postgresql.conf')",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT pg_stat_file('postgresql.conf')",
        "SELECT pg_ls_logdir()",
        "SELECT pg_ls_waldir()",
        "SELECT pg_ls_tmpdir()",
        "SELECT pg_ls_archive_statusdir()",
        "SELECT pg_current_logfile()",
        # SQLite: the DATABASE keyword is optional in the ATTACH grammar.
        "ATTACH '/etc/passwd' AS p",
        # MSSQL: OPENDATASOURCE is the connection-string counterpart of
        # OPENROWSET and can read remote/UNC files.
        "SELECT * FROM OPENDATASOURCE('SQLNCLI11', 'Server=remote').db.sys.tables",
    ],
)
def test_dangerous_pg_file_family_blocked(session, query):
    agent = _make_agent(session)
    rejection = agent._validate_query(query)
    assert rejection is not None
    assert "REJECTED" in rejection


def test_benign_pg_functions_not_blocked(session):
    """Non-disclosure pg_* functions must remain allowed (no over-match)."""
    agent = _make_agent(session)
    assert agent._validate_query("SELECT pg_typeof(1)") is None
    assert agent._validate_query("SELECT pg_backend_pid()") is None


def test_multi_statement_with_buried_drop_blocked(session):
    agent = _make_agent(session)
    rejection = agent._validate_query("SELECT 1; DROP TABLE items")
    assert rejection is not None
    assert "REJECTED" in rejection


def test_allow_dangerous_operations_bypasses_all_checks(session):
    agent = _make_agent(session, allow_dangerous_operations=True)
    poc = "DROP TABLE IF EXISTS log;\n" "COPY log(content) FROM PROGRAM 'id';\n"
    assert agent._validate_query(poc) is None
    assert agent._validate_query("DROP TABLE items") is None
    assert agent._validate_query("EXEC xp_cmdshell 'id'") is None


def test_extended_allowlist_permits_writes(session):
    agent = _make_agent(
        session,
        allowed_statement_types=["SELECT", "INSERT", "UPDATE", "DELETE"],
    )
    assert agent._validate_query("UPDATE items SET name='b' WHERE id=1") is None
    assert agent._validate_query("INSERT INTO items VALUES (2, 'b')") is None
    assert agent._validate_query("DELETE FROM items WHERE id=1") is None
    # Still blocks CREATE/DROP even with writes allowed.
    assert agent._validate_query("DROP TABLE items") is not None
    assert agent._validate_query("CREATE TABLE x (id int)") is not None
    # Still blocks dialect-specific dangerous primitives.
    assert agent._validate_query("SELECT load_extension('e')") is not None


# ---------------------------------------------------------------------------
# Integration tests via run_query (no LLM involved)
# ---------------------------------------------------------------------------


def test_run_query_rejects_drop_without_executing(session):
    agent = _make_agent(session)
    result = agent.run_query(RunQueryTool(query="DROP TABLE items"))
    assert "REJECTED" in result
    # The table must still exist after the rejected call.
    rows = session.execute(
        __import__("sqlalchemy").text("SELECT COUNT(*) FROM items")
    ).scalar()
    assert rows == 1


def test_run_query_allows_select(session):
    agent = _make_agent(session)
    result = agent.run_query(RunQueryTool(query="SELECT name FROM items"))
    assert "REJECTED" not in result
    assert "a" in result


def test_run_query_with_dangerous_ops_allowed_runs_drop(session):
    agent = _make_agent(session, allow_dangerous_operations=True)
    result = agent.run_query(RunQueryTool(query="DROP TABLE items"))
    assert "REJECTED" not in result
    # Sanity check that the table actually got dropped.
    with pytest.raises(Exception):
        session.execute(
            __import__("sqlalchemy").text("SELECT COUNT(*) FROM items")
        ).scalar()


# ---------------------------------------------------------------------------
# Regex-blocklist bypass via quoting / comments / schema qualification
# (GHSA-6xc5-4r68-67fc) -- caught by the AST-side function-name check.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        # Quoted identifier: the `"` between name and `(` defeats `\bpg_..\s*\(`.
        "SELECT \"pg_read_file\"('/etc/passwd')",
        # Inline comment between name and `(`.
        "SELECT pg_read_file/**/('/etc/passwd')",
        # Schema-qualified + quoted.
        "SELECT pg_catalog.\"pg_read_file\"('/etc/passwd')",
        # Schema-qualified (unquoted) form -- equivalent function call.
        "SELECT pg_catalog.pg_read_file('/etc/passwd')",
        # Same tricks applied to the rest of the dangerous family.
        "SELECT \"pg_stat_file\"('x')",
        "SELECT pg_ls_logdir/**/()",
        "SELECT pg_catalog.lo_import('/etc/passwd')",
        # Case-folding: validator must be case-insensitive on the AST too.
        "SELECT PG_READ_FILE('/etc/passwd')",
        "SELECT Pg_Catalog.\"Pg_Read_File\"('x')",
    ],
)
def test_ast_dangerous_function_bypasses_blocked(session, query):
    """The reporter's regex bypasses (and equivalents) must be rejected by the
    AST-side function-name check."""
    agent = _make_agent(session)
    rejection = agent._validate_query(query)
    assert rejection is not None
    assert "REJECTED" in rejection


@pytest.mark.parametrize(
    "query",
    [
        # Benign pg_* functions outside the dangerous prefix set must remain
        # allowed (guard against AST-check over-match).
        "SELECT pg_typeof(1)",
        "SELECT pg_backend_pid()",
        # Quoted/schema-qualified forms of benign functions must also pass.
        'SELECT "pg_typeof"(1)',
        "SELECT pg_catalog.pg_backend_pid()",
        # Ordinary user query.
        "SELECT name FROM items",
    ],
)
def test_ast_check_does_not_overmatch_benign_functions(session, query):
    agent = _make_agent(session)
    assert agent._validate_query(query) is None


# ---------------------------------------------------------------------------
# Nested writes: the top-level parse node does not determine what a statement
# writes (GHSA-3gpx-vwr3-xvwx / GHSA-wc83-4cvx-p8xc).
# ---------------------------------------------------------------------------


NESTED_WRITE_QUERIES = [
    # Data-modifying CTEs: top node parses as Select, the CTE still executes.
    "WITH x AS (DELETE FROM items RETURNING *) SELECT count(*) FROM x",
    "WITH x AS (UPDATE items SET name='b' RETURNING *) SELECT * FROM x",
    "WITH x AS (INSERT INTO items VALUES (2,'b') RETURNING *) SELECT * FROM x",
    # Nested a level deeper, to be sure the walk is recursive.
    "WITH a AS (WITH b AS (DELETE FROM items RETURNING *) SELECT * FROM b)"
    " SELECT * FROM a",
    # SELECT ... INTO creates and populates a table; no nested Create node,
    # it is carried on the Select's `into` arg.
    "SELECT * INTO evil FROM items",
]


@pytest.mark.parametrize("query", NESTED_WRITE_QUERIES)
def test_nested_writes_blocked_under_select_only_default(session, query):
    agent = _make_agent(session)
    rejection = agent._validate_query(query)
    assert rejection is not None, f"nested write slipped through: {query}"
    assert "REJECTED" in rejection


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM items",
        "SELECT count(*) FROM items WHERE name = 'a'",
        # A read-only CTE must still be allowed -- the check keys on the
        # embedded statement type, not on the presence of a WITH clause.
        "WITH x AS (SELECT * FROM items) SELECT count(*) FROM x",
        "WITH a AS (SELECT 1 AS n), b AS (SELECT n FROM a) SELECT * FROM b",
    ],
)
def test_read_only_queries_still_allowed(session, query):
    agent = _make_agent(session)
    assert agent._validate_query(query) is None


def test_nested_write_allowed_when_operator_extends_allowlist(session):
    """The gate keys on `allowed_statement_types`, not on nesting itself."""
    agent = _make_agent(
        session,
        allowed_statement_types=["SELECT", "DELETE"],
    )
    query = "WITH x AS (DELETE FROM items RETURNING *) SELECT count(*) FROM x"
    assert agent._validate_query(query) is None
    # An embedded INSERT is still not in the extended allowlist.
    insert_q = (
        "WITH x AS (INSERT INTO items VALUES (2,'b') RETURNING *) SELECT * FROM x"
    )
    assert agent._validate_query(insert_q) is not None


def test_nested_write_bypass_when_dangerous_ops_allowed(session):
    agent = _make_agent(session, allow_dangerous_operations=True)
    query = "WITH x AS (DELETE FROM items RETURNING *) SELECT count(*) FROM x"
    assert agent._validate_query(query) is None


MERGE_QUERY = (
    "MERGE INTO items t USING items s ON t.id = s.id "
    "WHEN MATCHED THEN UPDATE SET name = s.name "
    "WHEN NOT MATCHED THEN INSERT (id, name) VALUES (s.id, s.name)"
)


def test_merge_actions_are_not_treated_as_nested_statements(session):
    """A MERGE's WHEN actions belong to the merge, not to the allowlist.

    sqlglot parses `WHEN MATCHED THEN UPDATE/INSERT/DELETE` as child write
    nodes. Counting them as nested statements would force an operator who
    allows MERGE to also allow standalone UPDATE/INSERT/DELETE.
    """
    agent = _make_agent(session, allowed_statement_types=["MERGE"])
    assert agent._validate_query(MERGE_QUERY) is None


def test_merge_still_blocked_when_not_allowlisted(session):
    agent = _make_agent(session)  # SELECT-only default
    rejection = agent._validate_query(MERGE_QUERY)
    assert rejection is not None
    assert "REJECTED" in rejection


def test_merge_nested_in_cte_is_still_caught(session):
    """Only the top-level MERGE's own actions are exempt."""
    agent = _make_agent(session)
    query = (
        "WITH x AS (MERGE INTO items t USING items s ON t.id = s.id "
        "WHEN MATCHED THEN UPDATE SET name = s.name RETURNING *) "
        "SELECT * FROM x"
    )
    rejection = agent._validate_query(query)
    assert rejection is not None
    assert "REJECTED" in rejection


# ---------------------------------------------------------------------------
# SQLite `fileio` file-I/O functions, callable from inside a plain SELECT
# (GHSA-vqhw-xhff-464q).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        # Plain call form. No raw-text pattern covers these names (see the
        # comment in `_DANGEROUS_SQL_PATTERNS`); every case here, including
        # the plain one, is caught by the AST function-name check.
        "SELECT readfile('/etc/passwd')",
        "SELECT writefile('/tmp/pwn', 'owned')",
        "SELECT writefile('/tmp/pwn', 'owned', 511)",
        "SELECT lsdir('/')",
        "SELECT edit('x')",
        # `fsdir` is table-valued: it appears in FROM, not in the projection.
        "SELECT * FROM fsdir('/')",
        # Quoted identifier: the `"` between name and `(` would defeat a
        # `\bname\s*\(` pattern, which is part of why the check is on the AST.
        "SELECT \"readfile\"('/etc/passwd')",
        "SELECT \"writefile\"('/tmp/pwn', 'owned')",
        # Inline comment between name and `(`.
        "SELECT readfile/**/('/etc/passwd')",
        "SELECT writefile/**/('/tmp/pwn', 'owned')",
        # Schema qualification.
        "SELECT main.readfile('/etc/passwd')",
        "SELECT main.\"writefile\"('/tmp/pwn', 'owned')",
        # Case folding.
        "SELECT READFILE('/etc/passwd')",
        "SELECT WriteFile('/tmp/pwn', 'owned')",
        # Buried in a subquery / CTE, still a SELECT at the top level.
        "WITH x AS (SELECT writefile('/tmp/pwn', 'owned') AS n) SELECT * FROM x",
        "SELECT (SELECT readfile('/etc/passwd')) AS leaked",
        "SELECT name FROM items WHERE readfile('/etc/passwd') IS NOT NULL",
    ],
)
def test_sqlite_file_io_functions_blocked(session, query):
    """SQLite `fileio` file-I/O functions must be rejected under the
    SELECT-only default (GHSA-vqhw-xhff-464q).

    `writefile` performs a filesystem write from inside a plain SELECT, so
    without this check the `allowed_statement_types` allowlist classifies the
    query as SELECT and authorizes a write it never intended to permit.
    """
    agent = _make_agent(session)
    rejection = agent._validate_query(query)
    assert rejection is not None, f"SQLite file-I/O slipped through: {query}"
    assert "REJECTED" in rejection


def test_sqlite_file_io_still_blocked_with_writes_allowed(session):
    """Extending `allowed_statement_types` must not unblock file I/O."""
    agent = _make_agent(
        session,
        allowed_statement_types=["SELECT", "INSERT", "UPDATE", "DELETE"],
    )
    assert agent._validate_query("SELECT readfile('/etc/passwd')") is not None
    assert agent._validate_query("SELECT writefile('/tmp/pwn', 'x')") is not None


def test_sqlite_file_io_bypassed_by_dangerous_ops_flag(session):
    """`allow_dangerous_operations=True` remains a full bypass, by design."""
    agent = _make_agent(session, allow_dangerous_operations=True)
    assert agent._validate_query("SELECT readfile('/etc/passwd')") is None


@pytest.mark.parametrize(
    "query",
    [
        # SQLite lets a table-valued function take its arguments through
        # hidden-column constraints instead of a call, so this reads
        # /etc/passwd while parsing as an ordinary table reference -- no `(`
        # after the name, and the AST node is a Table, not a Func.
        # https://sqlite.org/vtab.html#tabfunc2
        "SELECT data FROM fsdir WHERE path = '/etc/passwd'",
        "SELECT length(data) FROM fsdir WHERE path = 'pyproject.toml'",
        "SELECT name FROM lsdir WHERE path = '/'",
        # Schema-qualified form of the same.
        "SELECT data FROM main.fsdir WHERE path = '/etc/passwd'",
        # Quoted identifier.
        "SELECT data FROM \"fsdir\" WHERE path = '/etc/passwd'",
        # Case folding, and an alias to hide the name downstream.
        "SELECT f.data FROM FSDIR f WHERE f.path = '/etc/passwd'",
        # Joined onto a legitimate table, and buried in a subquery.
        "SELECT i.name, f.data FROM items i, fsdir f WHERE f.path = '/etc/passwd'",
        "SELECT (SELECT data FROM fsdir WHERE path = '/etc/passwd') AS leaked",
        # A decoy CTE must not launder the reference. A CTE cannot be
        # schema-qualified, so `main.fsdir` still addresses the virtual table
        # even while a CTE named `fsdir` is in scope.
        "WITH fsdir(x) AS (SELECT 1) "
        "SELECT length(data) FROM main.fsdir WHERE path = 'pyproject.toml'",
        # ...and a CTE confined to a subquery does not shadow the outer
        # reference, which still resolves to the virtual table.
        "SELECT length(data) FROM fsdir WHERE path = 'pyproject.toml' "
        "AND EXISTS (WITH fsdir AS (SELECT 1) SELECT * FROM fsdir)",
        # ...nor does a quoted CTE name that only *looks* the same: SQLite
        # keeps the spaces, so `" fsdir "` is a different object and the
        # reference still reaches the virtual table.
        'WITH " fsdir "(x) AS (SELECT 1) '
        "SELECT data FROM fsdir WHERE path = '/etc/passwd'",
        # An empty quoted schema is still a qualifier, so the decoy CTE must
        # not exempt it either.
        'WITH fsdir(x) AS (SELECT 1) SELECT data FROM "".fsdir '
        "WHERE path = '/etc/passwd'",
        # Unqualified-looking but qualified forms with no CTE at all.
        "SELECT data FROM \"\".fsdir WHERE path = '/etc/passwd'",
        "SELECT data FROM temp.fsdir WHERE path = '/etc/passwd'",
    ],
)
def test_sqlite_table_valued_file_io_blocked(session, query):
    """A table-valued file-I/O function used as a bare table must be rejected
    (GHSA-vqhw-xhff-464q)."""
    agent = _make_agent(session)
    rejection = agent._validate_query(query)
    assert rejection is not None, f"table-valued file-I/O slipped through: {query}"
    assert "REJECTED" in rejection


@pytest.mark.parametrize(
    "query",
    [
        # Guard against over-match: a column or alias merely *named* like one
        # of the blocked functions is not a call and must stay allowed.
        "SELECT name AS readfile FROM items",
        "SELECT name AS writefile FROM items",
        # A string literal that merely contains a blocked name is not a call.
        # (A raw-text `name\\s*\\(` pattern would reject these.)
        "SELECT 'writefile(x)'",
        "SELECT 'readfile(/etc/passwd)' AS note",
        "SELECT name FROM items WHERE name <> 'edit('",
        # A CTE shadows a same-named table-valued function, so this reads the
        # CTE and touches no filesystem.
        "WITH fsdir(x) AS (SELECT 1) SELECT * FROM fsdir",
        "WITH lsdir AS (SELECT 1 AS x) SELECT x FROM lsdir",
        # The CTE is still in scope for a reference nested inside the query it
        # is attached to, so this reads the CTE too.
        "WITH fsdir(x) AS (SELECT 1) "
        "SELECT (SELECT count(*) FROM fsdir) AS n FROM items",
        # A CTE declared inside a subquery shadows the reference *within* that
        # subquery.
        "SELECT name FROM items WHERE EXISTS "
        "(WITH fsdir AS (SELECT 1 AS x) SELECT x FROM fsdir)",
        # Ordinary reads must be unaffected.
        "SELECT name FROM items WHERE name = 'a'",
        "SELECT count(*) FROM items",
    ],
)
def test_sqlite_file_io_check_does_not_overmatch(session, query):
    agent = _make_agent(session)
    assert agent._validate_query(query) is None


def test_run_query_rejects_writefile_without_executing(session, tmp_path):
    """End-to-end through `run_query`: no file may be created."""
    target = tmp_path / "pwn.txt"
    agent = _make_agent(session)
    result = agent.run_query(RunQueryTool(query=f"SELECT writefile('{target}', 'x')"))
    assert "REJECTED" in result
    assert not target.exists()


# ---------------------------------------------------------------------------
# `_nested_write_kinds` unit tests. The agent fixture is SQLite-backed, so
# dialect-specific parse shapes are exercised against the helper directly with
# an explicit dialect rather than through a fake agent.
# ---------------------------------------------------------------------------

_KIND_MAP = {
    sqlglot_exp.Select: "SELECT",
    sqlglot_exp.Insert: "INSERT",
    sqlglot_exp.Update: "UPDATE",
    sqlglot_exp.Delete: "DELETE",
    sqlglot_exp.Merge: "MERGE",
    sqlglot_exp.Create: "CREATE",
    sqlglot_exp.Drop: "DROP",
    sqlglot_exp.Alter: "ALTER",
    sqlglot_exp.TruncateTable: "TRUNCATE",
    sqlglot_exp.Command: "COMMAND",
}


def _kinds(query: str, dialect: str) -> set:
    return _nested_write_kinds(sqlglot.parse(query, read=dialect)[0], _KIND_MAP)


@pytest.mark.parametrize(
    "dialect, query, expected",
    [
        # MySQL `SELECT ... INTO @var` assigns a user variable and writes
        # nothing; sqlglot models it as Table(this=Parameter(...)). Treating it
        # as a table creation would reject a legitimate read under the
        # SELECT-only default.
        ("mysql", "SELECT name INTO @x FROM items", set()),
        # A real table target still counts.
        ("postgres", "SELECT * INTO evil FROM items", {"CREATE"}),
        ("tsql", "SELECT * INTO evil FROM items", {"CREATE"}),
        # `into` on a branch of a set operation: the top node is a Union, so
        # checking only the top-level Select would miss the table creation.
        (
            "tsql",
            "SELECT 1 AS x INTO new_t UNION ALL SELECT 2",
            {"CREATE"},
        ),
        # Ordinary reads.
        ("postgres", "SELECT * FROM items", set()),
        ("postgres", "WITH x AS (SELECT 1) SELECT * FROM x", set()),
        # Nested DML under a CTE.
        (
            "postgres",
            "WITH x AS (DELETE FROM items RETURNING *) SELECT * FROM x",
            {"DELETE"},
        ),
        # MERGE actions belong to the merge, not to the allowlist.
        (
            "postgres",
            "MERGE INTO a USING b ON a.id = b.id "
            "WHEN MATCHED THEN UPDATE SET a.n = b.n",
            set(),
        ),
        # ...but only the WHEN ... THEN action itself is exempt. A write in the
        # WHEN *condition* is a separate statement and must still be reported,
        # otherwise the exemption becomes a smuggling route.
        (
            "postgres",
            "MERGE INTO a USING b ON a.id = b.id "
            "WHEN MATCHED AND EXISTS ("
            "WITH x AS (DELETE FROM c RETURNING *) SELECT * FROM x"
            ") THEN UPDATE SET a.n = b.n",
            {"DELETE"},
        ),
        # A MERGE nested inside a CTE is still reported, as MERGE.
        (
            "postgres",
            "WITH x AS (MERGE INTO a USING b ON a.id = b.id "
            "WHEN MATCHED THEN UPDATE SET a.n = b.n RETURNING *) SELECT * FROM x",
            {"MERGE"},
        ),
    ],
)
def test_nested_write_kinds_across_dialects(dialect, query, expected):
    assert _kinds(query, dialect) == expected
