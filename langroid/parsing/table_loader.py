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
            # Read the data
        data = pd.read_csv(path_or_url, sep=sep)

        return data

    except Exception as e:
        raise ValueError(
            "Unable to read data. "
            "Please ensure it is correctly formatted. Error: " + str(e)
        )
