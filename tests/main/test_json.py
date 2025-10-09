import json

import pytest

from langroid.parsing.parse_json import (
    extract_top_level_json,
    parse_imperfect_json,
    top_level_json_field,
)


@pytest.mark.parametrize(
    "s, expected",
    [
        ("nothing to see here", []),
        (
            '{\n"key": \n"value \n with unescaped \nnewline"\n}',
            ['{"key": "value \\n with unescaped \\nnewline"}'],
        ),
        (
            '{\n"key": \n"value \\n with escaped \\nnewline"}',
            ['{"key": "value \\n with escaped \\nnewline"}'],
        ),
        (
            """
            Ok, thank you.
            {
                "request": "file_exists",
                "filename": "test.txt"
            }
            Hope you can tell me!
        """,
            [
                """
            {
                "request": "file_exists",
                "filename": "test.txt"
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
        # The below case has lots of json headaches/failures:
        # trailing commans and forgotten quotes
        (
            """
            {
            key_no_quotes: "value",
            "key": value_no_quote,
            key1: value with spaces,
            key2: 24,
            key3: { "a": b, "c": d e, 
               "f": g h k,
               }, },
            """,
            [
                """
                {
                "key_no_quotes": "value",
                "key": "value_no_quote",
                "key1": "value with spaces",
                "key2": 24,
                "key3": {"a": "b", "c": "d e", "f": "g h k"}
                }
                """
            ],
        ),
    ],
)
def test_extract_top_level_json(s, expected):
    top_level_jsons = extract_top_level_json(s)
    top_level_jsons = [json.loads(s.replace("'", '"')) for s in top_level_jsons]
    expected = [json.loads(s.replace("'", '"')) for s in expected]
    assert len(top_level_jsons) == len(expected)
    assert top_level_jsons == expected


@pytest.mark.parametrize(
    "input_json,expected_output",
    [
        # TODO - this aspect of parse_imperfect_json is NOT used anywhere --
        # if we do want to use it, how do we rationalize this behavior?
        (
            '{"key": "value \n with unescaped \nnewline"}',
            {"key": "value \n with unescaped \nnewline"},
        ),
        (
            '{"key": "value \\n with escaped \\nnewline"}',
            {"key": "value \n with escaped \nnewline"},
        ),
        ('{"key": "value", "number": 42}', {"key": "value", "number": 42}),
        (
            '{"key": "value", "number": 42,}',
            {"key": "value", "number": 42},
        ),  # extra comma
        ('{"key": null}', {"key": None}),
        ('{"t": true, "f": false}', {"t": True, "f": False}),
        ("{'key': 'value'}", {"key": "value"}),
        ("{'key': (1, 2, 3)}", {"key": (1, 2, 3)}),
        ("{key: 'value'}", {"key": "value"}),
        ("{'key': value}", {"key": "value"}),
        ("{key: value}", {"key": "value"}),
        (
            '{"key": "you said "hello" yesterday"}',  # did not escape inner quotes
            {"key": 'you said "hello" yesterday'},
        ),
        ("[1, 2, 3]", [1, 2, 3]),
        (
            """
    {
        "string": "Hello, World!",
        "number": 42,
        "float": 3.14,
        "boolean": true,
        "null": null,
        "array": [1, 2, 3],
        "object": {"nested": "value"},
        "mixed_array": [1, "two", {"three": 3}]
    }
    """,
            {
                "string": "Hello, World!",
                "number": 42,
                "float": 3.14,
                "boolean": True,
                "null": None,
                "array": [1, 2, 3],
                "object": {"nested": "value"},
                "mixed_array": [1, "two", {"three": 3}],
            },
        ),
    ],
)
def test_parse_imperfect_json(input_json, expected_output):
    assert parse_imperfect_json(input_json) == expected_output


@pytest.mark.parametrize(
    "invalid_input",
    [
        "",
        "not a json string",
        "True",  # This is a valid Python literal, but not a dict or list
        "42",  # This is a valid Python literal, but not a dict or list
    ],
)
def test_invalid_json_raises_error(invalid_input):
    with pytest.raises(ValueError):
        parse_imperfect_json(invalid_input)


@pytest.mark.parametrize(
    "s, field, expected",
    [
        # Scalar JSON should return "" (no crash)
        ("{1}", "recipient", ""),
        ('{"a": 1}', "a", 1),
        # Dict with field
        ('{"recipient": "Alice"}', "recipient", "Alice"),
        # List of dicts
        ('[{"recipient": "Bob"}]', "recipient", "Bob"),
        # Mixed text with dict
        ('Some text {"recipient": "Charlie"} more text', "recipient", "Charlie"),
        # Field not found
        ('{"other": "value"}', "recipient", ""),
    ],
)
def test_top_level_json_field(s, field, expected):
    assert top_level_json_field(s, field) == expected


def test_top_level_json_field_never_crashes():
    """Test that top_level_json_field never crashes with malformed inputs."""
    # Test cases that should not crash, just return ""
    malformed_inputs = [
        "",  # Empty string
        "not json at all",  # No JSON
        "{broken json",  # Incomplete JSON
        '{"key": undefined}',  # JavaScript-style undefined (gets repaired)
        "{\"malformed\": 'quotes'}",  # Wrong quotes
        "}{",  # Backwards braces
        "{{{",  # Nested unclosed
        '{"key": null, "key2": }',  # Trailing comma with no value
        '{"recipient": }',  # Field exists but no value
    ]

    for malformed in malformed_inputs:
        # Should never crash, just return empty string or found value
        result = top_level_json_field(malformed, "recipient")
        assert isinstance(result, (str, int, float, bool, type(None)))
