import pytest

from langroid.parsing.utils import closest_string


@pytest.mark.parametrize(
    "query, string_list, expected",
    [
        ("Bat  ", ["cat ", " Bat", "rat", " Hat"], " Bat"),
        ("rat", ["cat ", " Bat", "rat", " Hat"], "rat"),
        ("no_match", ["cat ", " Bat", "rat", " Hat"], "No match found"),
        ("BAT  ", ["cat ", " Bat", "rat", " Hat"], " Bat"),
    ],
)
def test_closest_string(query, string_list, expected):
    assert closest_string(query, string_list) == expected
