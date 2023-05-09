from llmagent.parsing.json import extract_top_level_json
import json
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
        ),  # should not recognize array as json
    ],
)
def test_extract_top_level_json(s, expected):
    top_level_jsons = extract_top_level_json(s)
    top_level_jsons = [json.loads(s.replace("'", '"')) for s in top_level_jsons]
    expected = [json.loads(s.replace("'", '"')) for s in expected]
    assert len(top_level_jsons) == len(expected)
    assert top_level_jsons == expected
