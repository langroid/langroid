import csv
from pathlib import Path

import pandas as pd
import pytest

from langroid.parsing.table_loader import read_tabular_data


@pytest.mark.parametrize("explicit_sep", [False, True])
@pytest.mark.parametrize(
    "separator, column",
    [
        (",", "gross, net"),
        (";", "gross; net"),
        ("\t", "gross\tnet"),
        (",", 'say "hello"'),
        (",", "owner's share"),
    ],
)
def test_read_quoted_headers(
    tmp_path: Path, explicit_sep: bool, separator: str, column: str
) -> None:
    path = tmp_path / "table.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=separator)
        writer.writerow([column, "cost"])
        writer.writerows([[10, 3], [20, 7]])

    result = read_tabular_data(str(path), separator if explicit_sep else None)

    pd.testing.assert_frame_equal(
        result, pd.DataFrame({column: [10, 20], "cost": [3, 7]})
    )


@pytest.mark.parametrize("separator", [",", ";", "\t"])
def test_read_tabular_data_skips_blank_headers(tmp_path: Path, separator: str) -> None:
    path = tmp_path / "table.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=separator)
        writer.writerow(["name", "", "amount", ""])
        writer.writerows([["Alice", "ignored", 10, "ignored"]])

    result = read_tabular_data(str(path), separator)

    pd.testing.assert_frame_equal(
        result, pd.DataFrame({"name": ["Alice"], "amount": [10]})
    )


def test_read_tabular_data_strips_column_whitespace(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    path.write_text(" name , amount \nAlice,10\n", encoding="utf-8")

    result = read_tabular_data(str(path), ",")

    pd.testing.assert_frame_equal(
        result, pd.DataFrame({"name": ["Alice"], "amount": [10]})
    )


def test_read_tabular_data_multichar_separator(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    path.write_text("name::amount\nAlice::10\n", encoding="utf-8")

    result = read_tabular_data(str(path), "::")

    pd.testing.assert_frame_equal(
        result, pd.DataFrame({"name": ["Alice"], "amount": [10]})
    )


@pytest.mark.parametrize("explicit_sep", [False, True])
@pytest.mark.parametrize("separator", [",", ";", "\t"])
@pytest.mark.parametrize("blank_headers", [False, True])
def test_read_tabular_data_preserves_duplicate_headers(
    tmp_path: Path, explicit_sep: bool, separator: str, blank_headers: bool
) -> None:
    path = tmp_path / "table.csv"
    headers = ["price", "price", "quantity"]
    rows = [[10, 20, 3], [30, 40, 5]]
    if blank_headers:
        headers = ["", "price", "", "price", "quantity", ""]
        rows = [[99, 10, 98, 20, 3, 97], [99, 30, 98, 40, 5, 97]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=separator)
        writer.writerow(headers)
        writer.writerows(rows)

    result = read_tabular_data(str(path), separator if explicit_sep else None)

    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame({"price": [10, 30], "price.1": [20, 40], "quantity": [3, 5]}),
    )


def test_read_tabular_data_preserves_duplicate_multichar_headers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "table.csv"
    path.write_text("price::::price::quantity\n10::99::20::3\n", encoding="utf-8")

    result = read_tabular_data(str(path), "::")

    pd.testing.assert_frame_equal(
        result, pd.DataFrame({"price": [10], "price.1": [20], "quantity": [3]})
    )


def test_read_tabular_data_preserves_existing_suffix_header(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    path.write_text("price,price.1,price\n10,20,30\n", encoding="utf-8")

    result = read_tabular_data(str(path), ",")

    pd.testing.assert_frame_equal(
        result, pd.DataFrame({"price": [10], "price.1": [20], "price.2": [30]})
    )
