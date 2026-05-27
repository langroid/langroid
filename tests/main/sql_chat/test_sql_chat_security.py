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

from langroid.agent.special.sql.sql_chat_agent import (
    SQLChatAgent,
    SQLChatAgentConfig,
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
