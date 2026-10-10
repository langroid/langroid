import json

import pytest

from langroid.parsing.parse_json import (
    _MAX_MULTILINE_INPUT_CHARS,
    _MAX_MULTILINE_QUOTES,
    extract_top_level_json,
    get_json_candidates,
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


@pytest.mark.parametrize("quote", ['"', "'"])
@pytest.mark.parametrize("braces", ["{", "}", "{}"])
def test_extract_top_level_json_preserves_multiline_string_braces(
    quote: str, braces: str
) -> None:
    value = f"first line\n{braces}\nlast line"
    tool_call = (
        f"{{{quote}request{quote}: {quote}test_tool{quote}, "
        f"{quote}value{quote}: {quote}{value}{quote}}}"
    )
    following_call = '{"request": "next_tool", "value": "complete"}'
    response = f"Tool calls:\n{tool_call}\n{following_call}\nPlease execute."

    extracted = extract_top_level_json(response)

    assert [json.loads(candidate) for candidate in extracted] == [
        {"request": "test_tool", "value": value},
        {"request": "next_tool", "value": "complete"},
    ]


@pytest.mark.parametrize("quote", ['"', "'"])
@pytest.mark.parametrize("separator", ["\n", " "])
def test_extract_top_level_json_keeps_unterminated_call_separate(
    quote: str, separator: str
) -> None:
    malformed_call = (
        f"{{{quote}request{quote}:{quote}bad{quote},"
        f"{quote}value{quote}:{quote}oops\n}}"
    )
    following_call = (
        f"{{{quote}request{quote}:{quote}good{quote},"
        f"{quote}value{quote}:{quote}ok{quote}}}"
    )
    response = malformed_call + separator + following_call

    assert get_json_candidates(response) == [malformed_call, following_call]
    assert [
        json.loads(candidate) for candidate in extract_top_level_json(response)
    ] == [
        {"request": "bad", "value": "oops"},
        {"request": "good", "value": "ok"},
    ]


@pytest.mark.parametrize("quote", ['"', "'"])
def test_extract_top_level_json_preserves_multiline_array_string(quote: str) -> None:
    value = f"first line\n{quote}quoted{quote} {{nested}}\\path\nlast line"
    escaped_value = value.replace("\\", "\\\\").replace(quote, "\\" + quote)
    tool_call = (
        f"{{{quote}request{quote}:{quote}test_tool{quote},"
        f"{quote}values{quote}:[{quote}{escaped_value}{quote}],"
        f"{quote}flag{quote}:true}}"
    )
    following_call = '{"request":"next_tool","value":"complete"}'
    response = tool_call + "\n" + following_call

    assert get_json_candidates(response) == [tool_call, following_call]
    assert [
        json.loads(candidate) for candidate in extract_top_level_json(response)
    ] == [
        {"request": "test_tool", "values": [value], "flag": True},
        {"request": "next_tool", "value": "complete"},
    ]


@pytest.mark.parametrize("quote", ['"', "'"])
@pytest.mark.parametrize("separator", ["\n", " "])
@pytest.mark.parametrize("suffix", [", John", ": note", "} text", "] text"])
def test_extract_top_level_json_keeps_mixed_quote_calls_separate(
    quote: str, separator: str, suffix: str
) -> None:
    other_quote = "'" if quote == '"' else '"'
    malformed_call = (
        f"{{{quote}request{quote}:{quote}bad{quote},"
        f"{quote}value{quote}:{quote}oops\n}}"
    )
    value = f"James{quote}{suffix}"
    following_call = (
        f"{{{other_quote}request{other_quote}:{other_quote}good{other_quote},"
        f"{other_quote}value{other_quote}:{other_quote}{value}{other_quote}}}"
    )
    response = malformed_call + separator + following_call

    assert get_json_candidates(response) == [malformed_call, following_call]
    assert [
        json.loads(candidate) for candidate in extract_top_level_json(response)
    ] == [
        {"request": "bad", "value": "oops"},
        {"request": "good", "value": value},
    ]


@pytest.mark.parametrize("quote", ['"', "'"])
def test_extract_top_level_json_preserves_multiline_opposite_quotes(
    quote: str,
) -> None:
    other_quote = "'" if quote == '"' else '"'
    value = (
        f"first line\nJames{other_quote}, John\n}}\n"
        f"{{{other_quote}request{other_quote}:"
        f"{other_quote}literal{other_quote}}}\nlast line"
    )
    tool_call = (
        f"{{{quote}request{quote}:{quote}test_tool{quote},"
        f"{quote}value{quote}:{quote}{value}{quote}}}"
    )
    following_call = '{"request":"next_tool","value":"complete"}'

    assert get_json_candidates(tool_call + "\n" + following_call) == [
        tool_call,
        following_call,
    ]
    assert [
        json.loads(candidate)
        for candidate in extract_top_level_json(tool_call + "\n" + following_call)
    ] == [
        {"request": "test_tool", "value": value},
        {"request": "next_tool", "value": "complete"},
    ]


@pytest.mark.parametrize("n", [2000, 4000, 8000, 16000])
def test_get_json_candidates_repeated_escaped_quotes(n: int) -> None:
    malformed_call = "{" + ('\\"' + "\n") * n + "}"
    following_call = '{"request":"good","value":"complete"}'

    assert get_json_candidates(malformed_call + "\n" + following_call) == [
        malformed_call,
        following_call,
    ]


def _multiline_call(total_chars: int) -> str:
    """One tool call whose value holds a raw newline and a bare brace.

    This is the shape #1203 recovers: the brace inside the value makes the
    single-line grammar treat it as structure and truncate the call. Padding
    goes inside the value, so the whole response is one candidate of the
    requested length and the quote count stays well under the quote budget.
    """
    prefix = '{"request": "write_file", "content": "first line\n{\n'
    suffix = '"}'
    pad = max(total_chars - len(prefix) - len(suffix), 0)
    return prefix + "x" * pad + suffix


# The speculative multiline scan costs a ~2.6x constant factor over the
# single-line grammar on input where `nested_expr` scans a long span without
# closing, so the absolute cost is held down by the input-size bound alone.
# These sizes are deliberately LITERAL rather than derived from
# `_MAX_MULTILINE_INPUT_CHARS`: a case that sizes itself from the constant
# moves with it and would stay green if the bound were raised back into the
# range where a truncated reply takes seconds (8 KiB -> 32 KiB took a 30 KB
# reply from 0.79s to 2.02s).
@pytest.mark.parametrize(
    "total_chars, recovered",
    [(8000, True), (9000, False)],
)
def test_multiline_recovery_is_bounded_by_input_size(
    total_chars: int, recovered: bool
) -> None:
    call = _multiline_call(total_chars)
    assert call.count('"') + call.count("'") <= _MAX_MULTILINE_QUOTES

    assert (get_json_candidates(call) == [call]) is recovered


def test_multiline_bounds_match_the_sizes_pinned_above() -> None:
    """Guards the literal sizes in the parametrization above.

    Those sizes only straddle the real boundary while the bound sits at
    8 KiB. If it is changed deliberately, this fails and points at the cases
    that need re-picking, rather than letting them silently stop testing the
    boundary.
    """
    assert _MAX_MULTILINE_INPUT_CHARS == 8 * 1024
    assert _MAX_MULTILINE_QUOTES == 512


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
            r'{"url": "https:\/\/example.org\/path"}',
            {"url": "https://example.org/path"},
        ),
        (r'{"text": "\ud83d\ude00"}', {"text": "😀"}),
        (
            r'{"\ud83d\ude00": {"url": "https:\/\/example.org"}}',
            {"😀": {"url": "https://example.org"}},
        ),
        (
            r'["https:\/\/example.org", "\ud83d\ude00"]',
            ["https://example.org", "😀"],
        ),
        (
            r'{"path": "C:\\work\\file.txt", "literal": "\\ud83d\\ude00"}',
            {"path": r"C:\work\file.txt", "literal": r"\ud83d\ude00"},
        ),
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
    "sloppy, expected",
    [
        # The shape langroid actually sees from weaker LLMs: a tool call with
        # unquoted values, followed by a nested object.
        (
            '{\n  request: foo,\n  args: bar baz,\n  opts: {"a": b, "c": d e}\n}',
            {"request": "foo", "args": "bar baz", "opts": {"a": "b", "c": "d e"}},
        ),
        (
            "{\n  request: run_query,\n  query: SELECT * FROM t,\n"
            '  filters: {"col": x, "op": eq}\n}',
            {
                "request": "run_query",
                "query": "SELECT * FROM t",
                "filters": {"col": "x", "op": "eq"},
            },
        ),
    ],
)
def test_repair_keeps_unquoted_value_fields(sloppy, expected):
    """Every key must survive repair, including after an unquoted value.

    json-repair 0.61.3 changed this: an unquoted value followed by another key
    and a nested object absorbs the next key/value pair into the value, so the
    field is silently dropped rather than parsed. That is a data-loss bug for
    us, since this is exactly the shape sloppy LLM tool calls take, and it is
    why `pyproject.toml` caps json-repair below 0.61.3. If this test starts
    failing after a dependency bump, the cap was lifted too far -- do not
    "fix" it by relaxing the assertion.
    """
    assert parse_imperfect_json(sloppy) == expected


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


@pytest.mark.parametrize(
    "following_call",
    [
        # the swallowed brace sits in an array element (after `[`)
        '{"request":"good","values":["James\'} text"]}',
        # ...in a later array element (after `,`)
        '{"request":"good","values":["ok","James\'} text"]}',
        # ...in a later object value (after `,` rather than `:`)
        '{"request":"good","value":"ok","note":"James\'} text"}',
    ],
)
def test_malformed_call_never_swallows_a_following_call(following_call: str) -> None:
    """A truncated call must not consume the next valid call's argument.

    Running a tool with another call's argument is worse than dropping the
    truncated one: dropping fails closed, so the agent hits
    `handle_llm_no_tool` and retries. The boundary guard therefore has to
    cover every context a quoted value can appear in, not just object values
    introduced by `:`.
    """
    malformed_call = "{'request':'bad','value':'oops\n}"
    response = malformed_call + "\n" + following_call

    assert get_json_candidates(response) == [malformed_call, following_call]
    extracted = [
        json.loads(candidate) for candidate in extract_top_level_json(response)
    ]
    assert extracted == [
        {"request": "bad", "value": "oops"},
        json.loads(following_call),
    ]
