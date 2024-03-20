from typing import Any

import pandas as pd


def stringify(x: Any) -> str:
    # Convert x to DataFrame if it is not one already
    if isinstance(x, pd.Series):
        df = x.to_frame()
    elif not isinstance(x, pd.DataFrame):
        return str(x)
    else:
        df = x

    # Truncate long text columns to 1000 characters
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda item: (
                    (item[:1000] + "...")
                    if isinstance(item, str) and len(item) > 1000
                    else item
                )
            )

    # Limit to 10 rows
    df = df.head(10)

    # Convert to string
    return df.to_string(index=False)  # type: ignore
