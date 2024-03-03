from typing import Any, Dict, List, Optional

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine


def extract_postgresql_descriptions(
    engine: Engine,
    multi_schema: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Extracts descriptions for tables and columns from a PostgreSQL database.

    This method retrieves the descriptions of tables and their columns
    from a PostgreSQL database using the provided SQLAlchemy engine.

    Args:
        engine (Engine): SQLAlchemy engine connected to a PostgreSQL database.
        multi_schema (bool): Generate descriptions for all schemas in the database.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping table names to a
        dictionary containing the table description and a dictionary of
        column descriptions.
    """
    inspector = inspect(engine)
    result: Dict[str, Dict[str, Any]] = {}

    def gen_schema_descriptions(schema: Optional[str] = None) -> None:
        table_names: List[str] = inspector.get_table_names(schema=schema)
        with engine.connect() as conn:
            for table in table_names:
                if schema is None:
                    table_name = table
                else:
                    table_name = f"{schema}.{table}"

                table_comment = (
                    conn.execute(
                        text(f"SELECT obj_description('{table_name}'::regclass)")
                    ).scalar()
                    or ""
                )

                columns = {}
                col_data = inspector.get_columns(table, schema=schema)
                for idx, col in enumerate(col_data, start=1):
                    col_comment = (
                        conn.execute(
                            text(
                                f"SELECT col_description('{table_name}'::regclass, "
                                f"{idx})"
                            )
                        ).scalar()
                        or ""
                    )
                    columns[col["name"]] = col_comment

                result[table_name] = {"description": table_comment, "columns": columns}

    if multi_schema:
        for schema in inspector.get_schema_names():
            gen_schema_descriptions(schema)
    else:
        gen_schema_descriptions()

    return result


def extract_mysql_descriptions(
    engine: Engine,
    multi_schema: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Extracts descriptions for tables and columns from a MySQL database.

    This method retrieves the descriptions of tables and their columns
    from a MySQL database using the provided SQLAlchemy engine.

    Args:
        engine (Engine): SQLAlchemy engine connected to a MySQL database.
        multi_schema (bool): Generate descriptions for all schemas in the database.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping table names to a
        dictionary containing the table description and a dictionary of
        column descriptions.
    """
    inspector = inspect(engine)
    result: Dict[str, Dict[str, Any]] = {}

    def gen_schema_descriptions(schema: Optional[str] = None) -> None:
        table_names: List[str] = inspector.get_table_names(schema=schema)

        with engine.connect() as conn:
            for table in table_names:
                if schema is None:
                    table_name = table
                else:
                    table_name = f"{schema}.{table}"

                query = text(
                    "SELECT table_comment FROM information_schema.tables WHERE"
                    " table_schema = :schema AND table_name = :table"
                )
                table_result = conn.execute(
                    query, {"schema": engine.url.database, "table": table_name}
                )
                table_comment = table_result.scalar() or ""

                columns = {}
                for col in inspector.get_columns(table, schema=schema):
                    columns[col["name"]] = col.get("comment", "")

                result[table_name] = {"description": table_comment, "columns": columns}

    if multi_schema:
        for schema in inspector.get_schema_names():
            gen_schema_descriptions(schema)
    else:
        gen_schema_descriptions()

    return result


def extract_default_descriptions(
    engine: Engine, multi_schema: bool = False
) -> Dict[str, Dict[str, Any]]:
    """Extracts default descriptions for tables and columns from a database.

    This method retrieves the table and column names from the given database
    and associates empty descriptions with them.

    Args:
        engine (Engine): SQLAlchemy engine connected to a database.
        multi_schema (bool): Generate descriptions for all schemas in the database.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping table names to a
        dictionary containing an empty table description and a dictionary of
        empty column descriptions.
    """
    inspector = inspect(engine)
    result: Dict[str, Dict[str, Any]] = {}

    def gen_schema_descriptions(schema: Optional[str] = None) -> None:
        table_names: List[str] = inspector.get_table_names(schema=schema)

        for table in table_names:
            columns = {}
            for col in inspector.get_columns(table):
                columns[col["name"]] = ""

            result[table] = {"description": "", "columns": columns}

    if multi_schema:
        for schema in inspector.get_schema_names():
            gen_schema_descriptions(schema)
    else:
        gen_schema_descriptions()

    return result


def extract_schema_descriptions(
    engine: Engine, multi_schema: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Extracts the schema descriptions from the database connected to by the engine.

    Args:
        engine (Engine): SQLAlchemy engine instance.
        multi_schema (bool): Generate descriptions for all schemas in the database.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary representation of table and column
        descriptions.
    """

    extractors = {
        "postgresql": extract_postgresql_descriptions,
        "mysql": extract_mysql_descriptions,
    }
    return extractors.get(engine.dialect.name, extract_default_descriptions)(
        engine, multi_schema=multi_schema
    )
