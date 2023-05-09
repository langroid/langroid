from llmagent.parsing.json import extract_top_level_json
import pytest


@pytest.mark.parametrize(
    "s, expected",
    [
        ("nothing to see here", []),
        (
            """
            Ok, thank you.
            {
                'request': 'file_exists',
                'filename': 'test.txt'
            }
            Hope you can tell me!
        """,
            [
                """
            {
                'request': 'file_exists',
                'filename': 'test.txt'
            }
            """
            ],
        ),
        (
            """
        [1, 2, 3]
        """,
            [],
        ),
    ],
)
def test_extract_top_level_json(s, expected):
    top_level_jsons = extract_top_level_json(s)
    assert len(top_level_jsons) == len(expected)
    assert top_level_jsons == expected
