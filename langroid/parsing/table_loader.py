from csv import Sniffer
from typing import List

import pandas as pd


def read_tabular_data(path_or_url: str, sep: None | str = None) -> pd.DataFrame:
    """
    Reads tabular data from a file or URL and returns a pandas DataFrame.
    The separator is auto-detected if not specified.

    Args:
        path_or_url (str): Path or URL to the file to be read.

    Returns:
        pd.DataFrame: Data from file or URL as a pandas DataFrame.

    Raises:
        ValueError: If the data cannot be read or is misformatted.
    """
    try:
        if sep is None:
            # Read the first few lines to guess the separator
            with pd.io.common.get_handle(path_or_url, "r") as file_handler:
                first_lines = "".join(file_handler.handle.readlines(5))
                sep = Sniffer().sniff(first_lines).delimiter
                # If it's a local file, reset to the beginning
                if hasattr(file_handler.handle, "seek"):
                    file_handler.handle.seek(0)

        # Read the data

        # get non-blank column names
        with pd.io.common.get_handle(path_or_url, "r") as f:
            header_line = f.handle.readline().strip()
            valid_cols = [col for col in header_line.split(sep) if col]
            valid_cols = [c.replace('"', "").replace("'", "") for c in valid_cols]
            if hasattr(f.handle, "seek"):
                f.handle.seek(0)

        # use only those columns
        data = pd.read_csv(path_or_url, sep=sep, usecols=valid_cols)
        data.columns = data.columns.str.strip()  # e.g. "  column 1  " -> "column 1"

        return data

    except Exception as e:
        raise ValueError(
            "Unable to read data. "
            "Please ensure it is correctly formatted. Error: " + str(e)
        )


def describe_dataframe(
    df: pd.DataFrame, filter_fields: List[str] = [], n_vals: int = 10
) -> str:
    """
    Generates a description of the columns in the dataframe,
    along with a listing of up to `n_vals` unique values for each column.
    Intended to be used to insert into an LLM context so it can generate
    appropriate queries or filters on the df.

    Args:
    df (pd.DataFrame): The dataframe to describe.
    filter_fields (list): A list of fields that can be used for filtering.
        When non-empty, the values-list will be restricted to these.
    n_vals (int): How many unique values to show for each column.

    Returns:
    str: A description of the dataframe.
    """
    description = []
    for column in df.columns.to_list():
        unique_values = df[column].dropna().unique()
        unique_count = len(unique_values)
        if column not in filter_fields:
            values_desc = f"{unique_count} unique values"
        else:
            if unique_count > n_vals:
                displayed_values = unique_values[:n_vals]
                more_count = unique_count - n_vals
                values_desc = f" Values - {displayed_values}, ... {more_count} more"
            else:
                values_desc = f" Values - {unique_values}"
        col_type = "string" if df[column].dtype == "object" else df[column].dtype
        col_desc = f"* {column} ({col_type}); {values_desc}"
        description.append(col_desc)

    all_cols = "\n".join(description)

    return f"""
        Name of each field, its type and unique values (up to {n_vals}):
        {all_cols}
        """
