from csv import Sniffer

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


def describe_dataframe(df: pd.DataFrame, sample_size: int = 5) -> str:
    """
    Generates a description of the columns in the dataframe, along with typical values.
    Intended to be used to insert into an LLM context so it can generate
    appropriate queries or filters on the df.

    Args:
        df (pd.DataFrame): The dataframe to describe.
        sample_size (int): The number of sample values to show for each column.

    Returns:
        str: A description of the dataframe.
    """
    description = []
    for column in df.columns:
        sample_values = df[column].dropna().head(sample_size).tolist()
        if len(sample_values) > 0 and isinstance(sample_values[0], str):
            # truncate to 100 chars
            sample_values = [v[:100] for v in sample_values]
        col_type = "string" if df[column].dtype == "object" else df[column].dtype
        col_desc = f"* {column} ({col_type}): {sample_values}"
        description.append(col_desc)

    all_cols = "\n".join(description)

    return f"""
        Name of each field, its type and some typical values:
        {all_cols}
        """
