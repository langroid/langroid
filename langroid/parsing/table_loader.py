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
