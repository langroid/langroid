from typing import Dict, Union

from sqlalchemy import MetaData


def populate_metadata_with_schema_tools(
    metadata: MetaData, info: Dict[str, Dict[str, Union[str, Dict[str, str]]]]
) -> Dict[str, Dict[str, Union[str, Dict[str, str]]]]:
    """
    Extracts information from an SQLAlchemy database's metadata and combines it
    with another dictionary with context descriptions.

    Args:
        metadata (MetaData): SQLAlchemy metadata object of the database.
        info (Dict[str, Dict[str, Any]]): A dictionary with table and column
                                             descriptions.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary with table and context information.
    """

    db_info: Dict[str, Dict[str, Union[str, Dict[str, str]]]] = {}

    # Create empty metadata dictionary with column datatypes
    for table_name, table in metadata.tables.items():
        # Populate tables with empty descriptions
        db_info[str(table_name)] = {
            "description": info[table_name]["description"] or "",
            "columns": {},
        }

        for column in table.columns:
            # Populate columns with datatype
            db_info[str(table_name)]["columns"][str(column.name)] = (  # type: ignore
                str(column.type)
            )

    return db_info


def populate_metadata(
    metadata: MetaData, info: Dict[str, Dict[str, Union[str, Dict[str, str]]]]
) -> Dict[str, Dict[str, Union[str, Dict[str, str]]]]:
    """
    Populate metadata based on the provided database metadata and additional info.

    Args:
        metadata (MetaData): Metadata object from SQLAlchemy.
        info (Dict): Additional information for database tables and columns.

    Returns:
        Dict: A dictionary containing populated metadata information.
    """

    # Fetch basic metadata info using available tools
    db_info: Dict[
        str, Dict[str, Union[str, Dict[str, str]]]
    ] = populate_metadata_with_schema_tools(metadata=metadata, info=info)

    # Iterate over tables to update column metadata
    for table_name in db_info.keys():
        # Update only if additional info for the table exists
        if table_name in info:
            for column_name in db_info[table_name]["columns"]:
                # Merge and update column description if available
                if column_name in info[table_name]["columns"]:
                    db_info[table_name]["columns"][column_name] = (  # type: ignore
                        db_info[table_name]["columns"][column_name]  # type: ignore
                        + "; "
                        + info[table_name]["columns"][column_name]  # type: ignore
                    )

    return db_info
