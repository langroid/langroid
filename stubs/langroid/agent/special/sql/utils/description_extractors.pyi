from typing import Any

from sqlalchemy.engine import Engine as Engine

from langroid.exceptions import LangroidImportError as LangroidImportError

def extract_postgresql_descriptions(
    engine: Engine, multi_schema: bool = False
) -> dict[str, dict[str, Any]]: ...
def extract_mysql_descriptions(
    engine: Engine, multi_schema: bool = False
) -> dict[str, dict[str, Any]]: ...
def extract_default_descriptions(
    engine: Engine, multi_schema: bool = False
) -> dict[str, dict[str, Any]]: ...
def extract_schema_descriptions(
    engine: Engine, multi_schema: bool = False
) -> dict[str, dict[str, Any]]: ...
