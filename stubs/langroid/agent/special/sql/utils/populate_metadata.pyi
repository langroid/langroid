from sqlalchemy import MetaData as MetaData

from langroid.exceptions import LangroidImportError as LangroidImportError

def populate_metadata_with_schema_tools(
    metadata: MetaData | list[MetaData],
    info: dict[str, dict[str, str | dict[str, str]]],
) -> dict[str, dict[str, str | dict[str, str]]]: ...
def populate_metadata(
    metadata: MetaData | list[MetaData],
    info: dict[str, dict[str, str | dict[str, str]]],
) -> dict[str, dict[str, str | dict[str, str]]]: ...
