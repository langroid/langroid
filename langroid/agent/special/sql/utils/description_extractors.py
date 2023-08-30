from typing import Any, Dict, List

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine


def extract_postgresql_descriptions(engine: Engine) -> Dict[str, Dict[str, Any]]:
    """
    Extracts descriptions for tables and columns from a PostgreSQL database.

    This method retrieves the descriptions of tables and their columns
    from a PostgreSQL database using the provided SQLAlchemy engine.

    Args:
        engine (Engine): SQLAlchemy engine connected to a PostgreSQL database.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping table names to a
        dictionary containing the table description and a dictionary of
        column descriptions.
    """
    inspector = inspect(engine)
    table_names: List[str] = inspector.get_table_names()

    result: Dict[str, Dict[str, Any]] = {}

    with engine.connect() as conn:
        for table in table_names:
            table_comment = (
                conn.execute(
                    text(f"SELECT obj_description('{table}'::regclass)")
                ).scalar()
                or ""
            )

            columns = {}
            col_data = inspector.get_columns(table)
            for idx, col in enumerate(col_data, start=1):
                col_comment = (
                    conn.execute(
                        text(f"SELECT col_description('{table}'::regclass, {idx})")
                    ).scalar()
                    or ""
                )
                columns[col["name"]] = col_comment

            result[table] = {"description": table_comment, "columns": columns}

    return result


def extract_mysql_descriptions(engine: Engine) -> Dict[str, Dict[str, Any]]:
    """Extracts descriptions for tables and columns from a MySQL database.

    This method retrieves the descriptions of tables and their columns
    from a MySQL database using the provided SQLAlchemy engine.

    Args:
        engine (Engine): SQLAlchemy engine connected to a MySQL database.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping table names to a
        dictionary containing the table description and a dictionary of
        column descriptions.
    """
    inspector = inspect(engine)
    table_names: List[str] = inspector.get_table_names()

    result: Dict[str, Dict[str, Any]] = {}

    with engine.connect() as conn:
        for table in table_names:
            query = text(
                "SELECT table_comment FROM information_schema.tables WHERE"
                " table_schema = :schema AND table_name = :table"
            )
            table_result = conn.execute(
                query, {"schema": engine.url.database, "table": table}
            )
            table_comment = table_result.scalar() or ""

            columns = {}
            for col in inspector.get_columns(table):
                columns[col["name"]] = col.get("comment", "")

            result[table] = {"description": table_comment, "columns": columns}

    return result


def extract_default_descriptions(engine: Engine) -> Dict[str, Dict[str, Any]]:
    """Extracts default descriptions for tables and columns from a database.

    This method retrieves the table and column names from the given database
    and associates empty descriptions with them.

    Args:
        engine (Engine): SQLAlchemy engine connected to a database.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping table names to a
        dictionary containing an empty table description and a dictionary of
        empty column descriptions.
    """
    inspector = inspect(engine)
    table_names: List[str] = inspector.get_table_names()

    result: Dict[str, Dict[str, Any]] = {}

    for table in table_names:
        columns = {}
        for col in inspector.get_columns(table):
            columns[col["name"]] = ""

        result[table] = {"description": "", "columns": columns}

    return result


def extract_schema_descriptions(engine: Engine) -> Dict[str, Dict[str, Any]]:
    """
    Extracts the schema descriptions from the database connected to by the engine.

    Args:
        engine (Engine): SQLAlchemy engine instance.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary representation of table and column
        descriptions.
    """

    extractors = {
        "postgresql": extract_postgresql_descriptions,
        "mysql": extract_mysql_descriptions,
    }
    return extractors.get(engine.dialect.name, extract_default_descriptions)(engine)
