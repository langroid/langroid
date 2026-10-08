"""Unit tests for the SQL schema-description extractors.

These drive the extractors against a recording fake connection, so they
need no database server, no network and no LLM credentials.
"""

from typing import Any, Dict, List, Optional

import pytest

from langroid.exceptions import LangroidImportError

try:
    from sqlalchemy import text
except ImportError as e:
    raise LangroidImportError(extra="sql", error=str(e))

from langroid.agent.special.sql.utils import description_extractors
from langroid.agent.special.sql.utils.description_extractors import (
    extract_postgresql_descriptions,
)

# A catalog name that closes the SQL string literal it used to be
# interpolated into, then appends a statement of the attacker's choosing.
# PostgreSQL permits any character inside a double-quoted identifier, so a
# user with CREATE TABLE can register exactly this name.
POISONED_TABLE = "pg_class'::regclass) UNION SELECT usename FROM pg_user--"


class _RecordingConnection:
    """Stands in for a DBAPI connection, recording what it is asked to run."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def execute(self, statement: Any, parameters: Any = None) -> Any:
        self.calls.append({"sql": str(statement), "params": parameters})

        class _Result:
            def scalar(self) -> Optional[str]:
                return None

        return _Result()

    def __enter__(self) -> "_RecordingConnection":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


class _FakeEngine:
    def __init__(self, conn: _RecordingConnection) -> None:
        self._conn = conn

    def connect(self) -> _RecordingConnection:
        return self._conn


class _FakeInspector:
    def __init__(self, table_names: List[str]) -> None:
        self._table_names = table_names

    def get_table_names(self, schema: Optional[str] = None) -> List[str]:
        return self._table_names

    def get_columns(
        self, table: str, schema: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return [{"name": "id"}]

    def get_schema_names(self) -> List[str]:
        return ["public"]


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> _RecordingConnection:
    """Run the PostgreSQL extractor over one poisoned catalog name."""
    conn = _RecordingConnection()
    monkeypatch.setattr(
        description_extractors,
        "inspect",
        lambda engine: _FakeInspector([POISONED_TABLE]),
    )
    extract_postgresql_descriptions(_FakeEngine(conn), multi_schema=False)
    assert conn.calls, "the extractor issued no queries; the test proves nothing"
    return conn


def test_postgresql_table_name_is_bound_not_interpolated(
    recorded: _RecordingConnection,
) -> None:
    """A quote in a catalog name must not reach the SQL text.

    `extract_postgresql_descriptions` built both of its queries with
    f-strings, so a table a low-privilege user could create was enough to
    run arbitrary SQL the moment `SQLChatAgent.__init__` read the catalog.
    """
    for call in recorded.calls:
        # the payload travels as a value...
        assert call["params"] is not None
        assert call["params"]["tbl"] == POISONED_TABLE
        # ...and never as SQL text
        assert POISONED_TABLE not in call["sql"]
        assert "UNION" not in call["sql"].upper()
        assert "pg_user" not in call["sql"]
        assert "'" not in call["sql"]
        assert ":tbl" in call["sql"]


def test_postgresql_extractor_queries_both_descriptions(
    recorded: _RecordingConnection,
) -> None:
    """Both description queries still run, and still carry the real name."""
    sqls = [call["sql"] for call in recorded.calls]
    assert any("obj_description" in sql for sql in sqls)
    assert any("col_description" in sql for sql in sqls)
    # the column index is bound too, rather than formatted in
    col_calls = [c for c in recorded.calls if "col_description" in c["sql"]]
    assert col_calls
    assert all(c["params"]["idx"] == 1 for c in col_calls)


def test_bind_parameter_name_survives_sqlalchemy_parsing() -> None:
    """`CAST(:tbl AS regclass)` is used because `:tbl::regclass` is not safe.

    SQLAlchemy reads the parameter in `:tbl::regclass` as `tb`, so binding
    `tbl` against it would not bind the value that gets sent.
    """
    assert sorted(
        text("SELECT obj_description(CAST(:tbl AS regclass))")._bindparams.keys()
    ) == ["tbl"]
    assert sorted(
        text("SELECT obj_description(:tbl::regclass)")._bindparams.keys()
    ) == ["tb"]
