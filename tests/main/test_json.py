import json
from typing import Any, Iterator

import pytest

from langroid.parsing import parse_json
from langroid.parsing.parse_json import (
    _MAX_MULTILINE_INPUT_CHARS,
    _MAX_MULTILINE_QUOTES,
    _SINGLE_LINE_CURLY_BRACES,
    _distinct_calls,
    _scan,
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
def test_extract_top_level_json_multiline_opposite_quotes_fails_closed(
    quote: str,
) -> None:
    """A value that quotes a whole object is read as separate calls.

    The multiline scan *could* reassemble this into one call whose `value`
    holds the inner object as text, and an earlier version of this test
    asserted exactly that. But the same input shape -- one candidate fully
    containing another self-contained call -- is how a malformed call swallows
    the next valid one and runs with its argument. Both readings of the text
    are legitimate, so extraction takes the safe one: separate calls, matching
    `main`'s behavior. The cost is that a tool argument which genuinely quotes
    a JSON object is no longer reassembled; that is the deliberate trade.

    What is still checked, as before: the opposite-quote value does not
    desynchronize the scan, each `}` inside the value is attributed to the
    right call, and the following well-formed call always survives intact.
    """
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
    inner_call = (
        f"{{{other_quote}request{other_quote}:" f"{other_quote}literal{other_quote}}}"
    )
    following_call = '{"request":"next_tool","value":"complete"}'
    truncated_first_call = (
        f"{{{quote}request{quote}:{quote}test_tool{quote},"
        f"{quote}value{quote}:{quote}first line\n"
        f"James{other_quote}, John\n}}"
    )

    assert get_json_candidates(tool_call + "\n" + following_call) == [
        truncated_first_call,
        inner_call,
        following_call,
    ]
    assert [
        json.loads(candidate)
        for candidate in extract_top_level_json(tool_call + "\n" + following_call)
    ] == [
        {"request": "test_tool", "value": f"first line\nJames{other_quote}, John"},
        {"request": "literal"},
        {"request": "next_tool", "value": "complete"},
    ]


def test_get_json_candidates_rejects_wholly_contained_call() -> None:
    """A valid call entirely inside a malformed one is not swallowed.

    The malformed call's unterminated single-quoted value spans past a
    complete double-quoted call. Reading it as one call would hand `bad` the
    argument belonging to `good`; `bad` must keep its own truncated value and
    `good` must survive as its own call.
    """
    response = (
        "{'request':'bad','value':'oops\n"
        "}\n"
        '{"request":"good","value":"ok"}\n'
        "'}"
    )

    assert get_json_candidates(response) == [
        "{'request':'bad','value':'oops\n}",
        '{"request":"good","value":"ok"}',
    ]
    assert [
        json.loads(candidate) for candidate in extract_top_level_json(response)
    ] == [
        {"request": "bad", "value": "oops"},
        {"request": "good", "value": "ok"},
    ]


def test_get_json_candidates_prefers_the_higher_yield_reading() -> None:
    """A contained call that is itself MALFORMED is not swallowed either.

    Identical to the wholly-contained case above except that `good` is missing
    its closing quote. Extraction repairs that happily, but it is not
    structurally self-contained, so the structural guard cannot see it -- and
    being merely repairable is too weak a signal to protect on, since almost
    any truncated fragment repairs into something.

    Yield is what separates them: reading this span as single-line calls
    dispatches two distinct calls (`bad` and `good`) where reading it as one
    multiline call dispatches one, so the two-call reading wins. On a genuine
    multiline recovery the count is one either way, and the tie keeps the
    recovery.
    """
    response = (
        "{'request':'bad','value':'oops\n" "}\n" '{"request":"good","value":"ok}\n' "'}"
    )

    assert get_json_candidates(response) == [
        "{'request':'bad','value':'oops\n}",
        '{"request":"good","value":"ok}',
    ]
    # Byte-identical to `main`: no regression that runs a tool with another
    # call's argument.
    assert extract_top_level_json(response) == [
        '{"request": "bad", "value": "oops"}',
        '{"request": "good", "value": "ok"}',
    ]


def test_get_json_candidates_higher_yield_reading_ignores_the_tool_name() -> None:
    """The same case, with both calls naming the SAME tool.

    A reply that calls one tool twice is where a swallow is most likely, and it
    is invisible to a count keyed on the tool name: both readings name
    `write_file` once, so such a count ties at 1 and the swallow survives --
    `write_file` would run with `ok` and the `oops` call would disappear.
    Identity is the whole repaired call, so the counts are 2 against 1 here.
    """
    response = (
        "{'request':'write_file','value':'oops\n"
        "}\n"
        '{"request":"write_file","value":"ok}\n'
        "'}"
    )

    # Byte-identical to `main`: both invocations survive, each with its own
    # argument.
    assert extract_top_level_json(response) == [
        '{"request": "write_file", "value": "oops"}',
        '{"request": "write_file", "value": "ok"}',
    ]


def test_distinct_calls_counts_dispatchable_calls_only() -> None:
    """The yield comparison counts dispatchable calls, keyed on the whole call.

    Only an object that repairs to a dict with a string `request` can dispatch
    a tool: without that restriction, ordinary JSON sitting inside a value
    would inflate the single-line reading and switch recovery off. And two
    calls to the same tool with different arguments are two calls, so the name
    alone cannot be the identity.
    """
    assert _distinct_calls(['{"request": "a"}', '{"request": "b"}']) == 2
    assert (
        _distinct_calls(['{"request": "a", "x": 1}', '{"request": "a", "x": 2}']) == 2
    )
    assert _distinct_calls(['{"request": "a"}', '{"request": "a"}']) == 1
    assert _distinct_calls(['{"x": 1}', "{not json at all}"]) == 0
    assert _distinct_calls(['{"request": 7}']) == 0
    assert _distinct_calls([]) == 0


@pytest.mark.parametrize(
    "literal, parsed", [("true", True), ("false", False), ("null", None)]
)
def test_get_json_candidates_protects_single_quoted_json_literals(
    literal: str,
    parsed: object,
) -> None:
    """A single-quoted contained call holding true/false/null is protected.

    The swallow guard can only protect a call it recognizes as whole, and
    these spellings fall between the loaders: `json.loads` rejects the single
    quotes, `ast.literal_eval` rejects the lowercase literals. Before
    `_is_self_contained_object` also tried YAML, such a call was missing from
    the `intact` set and was still swallowed -- `bad` ran with `good`'s
    arguments.
    """
    inner = "{'request':'good','value':'ok','flag':" + literal + "}"
    response = '{"request":"bad","value":"oops\n' "}\n" + inner + '\n"}'

    assert [
        json.loads(candidate) for candidate in extract_top_level_json(response)
    ] == [
        {"request": "bad", "value": "oops"},
        {"request": "good", "value": "ok", "flag": parsed},
    ]


def test_get_json_candidates_protects_contained_call_with_stray_bracket() -> None:
    """An unbalanced bracket does not cost a contained call its protection.

    Extraction repairs the `good` call, so the swallow guard has to see it as
    whole. Counting brackets as well as braces left its depth never
    returning to zero, which put it back in reach of the malformed call.
    """
    response = (
        "{'request':'bad','value':'oops\n"
        "}\n"
        '{"request":"good","value":1 # [\n'
        "}\n"
        "'}"
    )

    assert [
        json.loads(candidate) for candidate in extract_top_level_json(response)
    ] == [
        {"request": "bad", "value": "oops"},
        {"request": "good", "value": 1},
    ]


def test_get_json_candidates_falls_back_when_multiline_scan_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed multiline scan costs nothing the plain grammar would find.

    Both grammars are recursive and the multiline one spends more stack per
    position, so deeply nested input can exhaust it where the single-line one
    copes -- and `_scan` reports that as no matches at all, which dropped
    every call in a 527-character reply well inside both recovery bounds.

    The trigger is how much stack is left, which a test cannot control: the
    depth window moves with the caller, and under pytest both grammars
    happen to fail together, so a depth-based test would assert nothing here
    and break elsewhere. Hence the narrowest possible substitute -- the
    multiline grammar itself is replaced by one that raises -- so the
    fallback is exercised deterministically on every machine.
    """
    response = '{"request":"good","value":"a\nb"}\n{"request":"next"}'
    single_line_reading = [
        text for text, _, _ in _scan(_SINGLE_LINE_CURLY_BRACES, response)
    ]
    assert single_line_reading, "fixture must be matchable by the plain grammar"

    class _Exhausted:
        def scan_string(self, _: str) -> Iterator[Any]:
            raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(parse_json, "_CURLY_BRACES", _Exhausted())
    assert parse_json.get_json_candidates(response) == single_line_reading


@pytest.mark.parametrize("inner_quote", ['"', "'"])
@pytest.mark.parametrize("outer_quote", ['"', "'"])
def test_get_json_candidates_protects_contained_call_with_raw_newline(
    outer_quote: str, inner_quote: str
) -> None:
    """A contained call whose own value holds a raw newline is protected.

    A raw newline in a quoted value is exactly what this module recovers, so
    a complete call containing one must still count as self-contained. All
    four quote pairings are pinned because the loader-by-loader versions of
    `_is_self_contained_object` passed some and failed others: strict JSON
    rejected the raw newline, `strict=False` JSON still rejected single
    quotes, and Python forbids a raw newline inside a single-quoted string,
    so each spelling had to be found separately by review. The structural
    check is blind to the pairing, and this test is what holds it that way.
    """
    iq, oq = inner_quote, outer_quote
    inner = f"{{{iq}request{iq}:{iq}good{iq},{iq}value{iq}:{iq}ok\nnext{iq}}}"
    response = (
        f"{{{oq}request{oq}:{oq}bad{oq},{oq}value{oq}:{oq}oops\n" f"}}\n{inner}\n{oq}}}"
    )

    assert [
        json.loads(candidate) for candidate in extract_top_level_json(response)
    ] == [
        {"request": "bad", "value": "oops"},
        {"request": "good", "value": "ok\nnext"},
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
        # ...in a value sitting on the line AFTER its `:`
        '{"request":"good","value":\n"James\'} text"}',
        # ...in the first KEY of a nested object (after `{`)
        '{"request":"good","value":{"James\'} text":"ok"}}',
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


@pytest.mark.parametrize("separator", [' " ', " '  ", ' text " ', ' "" " '])
def test_stray_quote_between_calls_does_not_swallow(separator: str) -> None:
    """A stray quote between two calls must not cost the second one.

    The line-local boundary check reasons about quote positions by scanning
    the candidate's last line, so an unbalanced quote ahead of the following
    call shifts every boundary it computes and the check stops firing. The
    span comparison in `get_json_candidates` is what covers this: it asks
    whether the accepted candidate ends part-way through a call that is valid
    on its own, which does not depend on quote alignment at all.
    """
    malformed_call = "{'request':'bad','value':'oops\n}"
    following_call = '{"request":"good","value":"James\'} text"}'
    response = malformed_call + separator + following_call

    extracted = [
        json.loads(candidate) for candidate in extract_top_level_json(response)
    ]
    assert {"request": "good", "value": "James'} text"} in extracted
