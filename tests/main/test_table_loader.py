import csv
from pathlib import Path

import pandas as pd
import pytest

from langroid.parsing.table_loader import read_tabular_data


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
@pytest.mark.parametrize("explicit_sep", [False, True])
@pytest.mark.parametrize("separator", [",", ";", "\t"])
@pytest.mark.parametrize("first_column", ["name", "gross, net", ""])
def test_read_tabular_data_utf8_bom(
    tmp_path: Path,
    encoding: str,
    explicit_sep: bool,
    separator: str,
    first_column: str,
) -> None:
    path = tmp_path / "table.csv"
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.writer(handle, delimiter=separator, quoting=csv.QUOTE_ALL)
        writer.writerow([first_column, "amount"])
        writer.writerows([["Alice", 10], ["Bob", 20]])

    result = read_tabular_data(str(path), separator if explicit_sep else None)

    expected = pd.DataFrame({"amount": [10, 20]})
    if first_column:
        expected.insert(0, first_column, ["Alice", "Bob"])
    pd.testing.assert_frame_equal(result, expected)


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
