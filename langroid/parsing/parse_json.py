import ast
import json
from datetime import datetime
from typing import Any, Dict, Iterator, List, Tuple, Union

import yaml
from json_repair import repair_json
from pyparsing import (
    FollowedBy,
    ParseException,
    QuotedString,
    nested_expr,
    one_of,
    original_text_for,
    quoted_string,
)


def is_valid_json(json_str: str) -> bool:
    """Check if the input string is a valid JSON.

    Args:
        json_str (str): The input string to check.

    Returns:
        bool: True if the input string is a valid JSON, False otherwise.
    """
    try:
        json.loads(json_str)
        return True
    except ValueError:
        return False


def flatten(nested_list) -> Iterator[str]:  # type: ignore
    """Flatten a nested list into a single list of strings"""
    for item in nested_list:
        if isinstance(item, (list, tuple)):
            for subitem in flatten(item):
                yield subitem
        else:
            yield item


def _is_complete_json_object(candidate: str, source: str, loc: int) -> bool:
    """Validate multiline boundaries without repairing or changing the candidate."""
    try:
        return isinstance(json.loads(candidate), dict)
    except ValueError:
        pass

    # A brace inside a following single-line value must not end this object,
    # even if the swallowed prefix happens to be valid after normalization.
    #
    # The quoted string the brace falls inside may be an object value (after
    # `:`), an array element (after `[`) or any later item in either (after
    # `,`). Checking only `:` let an array element swallow the whole following
    # call: for `{'request':'bad','value':'oops\n}` followed by
    # `{"request":"good","values":["James'} text"]}`, the candidate ended at
    # `James'}` and the valid `good` call disappeared. Over-rejecting here only
    # costs multiline recovery -- the candidate falls back to the single-line
    # grammar -- whereas under-rejecting runs one call with another's argument.
    end = loc + len(candidate) - 1
    line_start = max(loc, source.rfind("\n", 0, end) + 1)
    line_end = source.find("\n", end)
    line = source[line_start : line_end if line_end >= 0 else len(source)]
    for _, start, stop in quoted_string.scan_string(line):
        if start >= end - line_start:
            break
        if start < end - line_start < stop:
            # These four are every position a JSON string can open in: an
            # object key (after `{` or `,`), an object value (after `:`), or an
            # array element (after `[` or `,`). Enumerating them exhaustively
            # rather than case by case is deliberate -- each delimiter missed
            # here is another way for a truncated call to swallow the next one.
            #
            # Take the prefix from `source`, not `line`: a value may sit on the
            # line after its delimiter, leaving `line[:start]` all whitespace
            # and the punctuation on the previous line.
            if source[: line_start + start].rstrip().endswith((":", ",", "[", "{")):
                return False

    try:
        return isinstance(json.loads(candidate, strict=False), dict)
    except ValueError:
        pass

    # Normalize quoted tokens only for validation, accepting single quotes and
    # raw newlines. The original text is still returned for the existing repair.
    normalized = _NORMALIZED_QUOTED_STRINGS.transform_string(candidate)
    try:
        return isinstance(json.loads(normalized), dict)
    except ValueError:
        pass
    try:
        return isinstance(ast.literal_eval(normalized), dict)
    except (ValueError, SyntaxError):
        return False


_QUOTED_STRINGS = QuotedString(
    '"', esc_char="\\", multiline=True, unquote_results=False
) | QuotedString("'", esc_char="\\", multiline=True, unquote_results=False)
_NORMALIZED_QUOTED_STRINGS = _QUOTED_STRINGS.copy().set_parse_action(
    lambda tokens: json.dumps(tokens[0][1:-1])
)
_MULTILINE_CURLY_BRACES = original_text_for(
    nested_expr(
        "{",
        "}",
        ignore_expr=(_QUOTED_STRINGS + FollowedBy(one_of(": , } ]"))) | quoted_string,
    )
).add_condition(
    lambda source, loc, tokens: _is_complete_json_object(tokens[0], source, loc)
)
_SINGLE_LINE_CURLY_BRACES = original_text_for(nested_expr("{", "}"))
_CURLY_BRACES = _MULTILINE_CURLY_BRACES | _SINGLE_LINE_CURLY_BRACES
# Bound speculative multiline scans on malformed, large or quote-dense output.
# Outside this budget, retain the original single-line extraction behavior.
#
# The multiline grammar's `ignore_expr` is retried at every character position,
# so on input where `nested_expr` scans a long span without closing (a reply
# truncated mid-string, say) it costs a 2-4x constant factor over the
# single-line grammar. That factor is unavoidable within this approach, so the
# length bound is all that keeps the absolute cost finite, and at 8 KiB the cost
# is NOT small: measured at the bound on a reply truncated mid-string, this
# module takes 960 ms-1.1 s against the single-line grammar's 260-570 ms, so the
# extra work is on the order of 550-700 ms. 64 KiB allowed ~3s on a 63 KB reply.
# That cost is paid per call site, and `extract_top_level_json` is called from
# several uncached places per turn, so the bound is a ceiling on a real
# regression rather than a formality -- lower it before raising it.
#
# Well-formed input of any size is unaffected — it closes immediately and never
# enters the slow scan, and a reply truncated mid-string is the shape that pays.
# Multiline recovery is therefore best-effort, and applies to tool calls up to
# 8 KiB; larger ones keep the original (truncating) behavior rather than
# stalling extraction.
_MAX_MULTILINE_INPUT_CHARS = 8 * 1024
_MAX_MULTILINE_QUOTES = 512


def _dispatchable_calls(candidates: List[str]) -> int:
    """Count the tool calls that one reading of a span would dispatch.

    Each candidate is repaired exactly as extraction would repair it, and is
    counted if it yields an object at all. A missing `request` does NOT make
    an object undispatchable: `Agent._get_one_tool_message` infers the request
    when exactly one enabled tool matches the fields present, so requiring the
    field undercounts the single-line reading and lets a swallow through --

        {'request':'write_file','value':'oops
        }
        {"value":"ok}
        '}

    where the second object's text ended up inside `write_file`'s argument and
    `main` returns the two objects separately. Counting every repaired object
    was measured not to cost any recovery: over the 6000-case differential
    fuzz the number of replies this module recovers more calls from than
    `main` is identical either way.

    Nothing is deduplicated. Two readings of the same text tie because each
    yields ONE call, not because equal calls collapse, so collapsing buys the
    comparison nothing -- and it costs, twice over. Keyed on the `request`
    name it loses a second call to the same tool; keyed on the whole repaired
    object it still loses an identical repeat, as in

        {'request':'write_file','value':'oops
        }
        {"request":"write_file","value":"oops}
        '}

    where `main` dispatches `write_file` twice. A tool whose effect is
    cumulative runs once instead of twice, so a repeat is a call like any
    other and has to be counted like one.

    Args:
        candidates: The candidate object texts making up one reading of a span.

    Returns:
        The number of calls that reading dispatches.
    """
    calls = 0
    for candidate in candidates:
        repaired = try_repair_json_yaml(candidate)
        if repaired is None:
            continue
        try:
            obj = json.loads(repaired)
        except ValueError:
            continue
        if isinstance(obj, dict):
            calls += 1
    return calls


def _scan(expr: Any, s: str) -> List[Tuple[str, int, int]]:
    """Matches as (text, start, end), so spans can be compared."""
    try:
        # Properly convert nested lists to strings
        return [(tokens[0], start, end) for tokens, start, end in expr.scan_string(s)]
    except Exception:
        return []


def get_json_candidates(s: str) -> List[str]:
    """Get top-level JSON candidates, i.e. strings between curly braces."""
    if (
        len(s) > _MAX_MULTILINE_INPUT_CHARS
        or s.count('"') + s.count("'") > _MAX_MULTILINE_QUOTES
    ):
        return [text for text, _, _ in _scan(_SINGLE_LINE_CURLY_BRACES, s)]

    matches = _scan(_CURLY_BRACES, s)
    if not matches:
        # The grammars are recursive, and the multiline one spends more stack
        # per position, so deeply nested input can exhaust it where the
        # single-line one copes. `_scan` reports that as no matches at all,
        # which would silently drop every call in an otherwise fine reply
        # (80 levels of nesting in 527 characters, well inside both bounds).
        # A failed speculative scan must never cost what the plain grammar
        # would have found.
        return [text for text, _, _ in _scan(_SINGLE_LINE_CURLY_BRACES, s)]

    candidates = [text for text, _, _ in matches]
    if all(is_valid_json(text) for text in candidates):
        # Nothing was recovered speculatively, so there is nothing to second-
        # guess, and the common case pays for no extra scan.
        return candidates

    # A speculative multiline candidate covers text the single-line grammar
    # would have split. That is the point when a quoted value holds a raw
    # newline -- but it is wrong when the candidate ENDS part-way through a
    # call that is valid on its own, because then a tool runs with another
    # call's argument. Dropping a truncated call instead fails closed: the
    # agent hits `handle_llm_no_tool` and retries.
    #
    # Containment is the test, not just straddling. A self-contained call the
    # candidate's end cuts through has plainly been swallowed, but so has one
    # that sits wholly INSIDE the candidate: the two readings of such text --
    # one call whose argument happens to contain JSON, or two calls of which
    # the first is truncated -- are both legitimate, and we take the second,
    # so a tool can never run with another call's argument. That costs some
    # genuine recoveries (a tool argument that really does quote a whole JSON
    # object is no longer reassembled) and is the deliberate trade.
    #
    # Comparing spans this way is an invariant over the result, so it holds
    # however quote alignment was lost -- `_is_complete_json_object`'s
    # line-local check stays as an early reject, but it cannot see every such
    # way. A candidate that IS the intact match (identical span) is of course
    # not swallowing itself.
    #
    # Which reading wins is decided by YIELD alone, and not by any structural
    # property of the contained text, because a contained call can be
    # swallowed while being malformed itself -- `good` below is missing its
    # closing quote, so it is not a structurally whole object, yet extraction
    # repairs it happily and the single-line reading recovers both calls:
    #
    #     {'request':'bad','value':'oops
    #     }
    #     {"request":"good","value":"ok}
    #     '}
    #
    # Being merely REPAIRABLE is far too weak a signal to admit into the
    # protected set on its own -- almost any truncated fragment repairs into
    # something, so treating repairable text as a call to protect would switch
    # multiline recovery off in general. What separates the two is yield; see
    # the condition below.
    single_line = _scan(_SINGLE_LINE_CURLY_BRACES, s)

    # Fall back per candidate, not for the whole response: a legitimate
    # multiline recovery earlier in the reply must survive a swallow later in
    # it. For a candidate that cut a call, emit the single-line reading of its
    # span instead, and skip anything that reading already covered.
    result: List[str] = []
    position = 0
    for text, start, end in matches:
        if end <= position:
            continue
        contained = [
            m
            for m in single_line
            if m[1] < end < m[2]  # the candidate's end cuts this call
            or (start < m[1] and m[2] <= end)  # this call sits wholly inside
        ]
        span_reading = [m[0] for m in single_line if start <= m[1] < end]
        # Compare yields: read the candidate's span both ways, repair each
        # piece, and prefer the single-line reading only when it dispatches
        # strictly MORE calls. A swallow shows up as 2 against 1, because the
        # swallowed call is lost entirely. A tie keeps the multiline recovery,
        # which is what a genuine recovery looks like -- one call read either
        # way -- so recovery survives. Being a comparison of outcomes and not
        # of spellings, it is blind to however the swallowed call happens to
        # be malformed.
        if contained and _dispatchable_calls(span_reading) > _dispatchable_calls(
            [text]
        ):
            for text_i, start_i, end_i in single_line:
                if start <= start_i < end and end_i > position:
                    result.append(text_i)
                    position = end_i
        else:
            result.append(text)
            position = end
    return result


def parse_imperfect_json(json_string: str) -> Union[Dict[str, Any], List[Any]]:
    if not json_string.strip():
        raise ValueError("Empty string is not valid JSON")

    # Preserve JSON escape semantics before trying Python literals or repairs.
    try:
        result = json.loads(json_string)
        if isinstance(result, (dict, list)):
            return result
    except json.JSONDecodeError:
        pass

    # Accept Python literals such as single quotes, True, and None.
    try:
        result = ast.literal_eval(json_string)
        if isinstance(result, (dict, list)):
            return result
    except (ValueError, SyntaxError):
        pass

    # If ast.literal_eval fails or returns non-dict/list, try repair_json
    json_repaired_obj = repair_json(json_string, return_objects=True)
    if isinstance(json_repaired_obj, (dict, list)):
        return json_repaired_obj
    else:
        try:
            # fallback on yaml
            yaml_result = yaml.safe_load(json_string)
            if isinstance(yaml_result, (dict, list)):
                return yaml_result
        except yaml.YAMLError:
            pass

    # If all methods fail, raise ValueError
    raise ValueError(f"Unable to parse as JSON: {json_string}")


def try_repair_json_yaml(s: str) -> str | None:
    """
    Attempt to load as json, and if it fails, try repairing the JSON.
    If that fails, replace any \n with space as a last resort.
    NOTE - replacing \n with space will result in format loss,
    which may matter in generated code (e.g. python, toml, etc)
    """
    # Nothing that already parses should be handed to a repairer. `strict=False`
    # is what admits the raw newline inside a string that multiline recovery
    # exists to preserve, and `repair_json` silently rewrites exactly that
    # content: on `{"request":"write_file","content":"first\n{\n"}` it drops the
    # trailing newline, and on config-like values it can read a line as another
    # field. Since the point of recovering such a call is to dispatch it with
    # the argument the model actually wrote, the parse result is serialized
    # directly and the repair path is left for text that genuinely needs it.
    try:
        parsed = json.loads(s, strict=False)
        if isinstance(parsed, (dict, list)):
            return json.dumps(parsed)
    except ValueError:
        pass

    # The same argument for the single-quoted spelling, which `json.loads`
    # rejects outright however lax `strict` is. `_NORMALIZED_QUOTED_STRINGS`
    # re-encodes each quoted token with `json.dumps`, which turns a raw newline
    # into `\n` and the quotes into double quotes.
    #
    # Only when the text carries no backslash at all, though: the normalizer
    # reads the bytes between the quotes literally, so it would re-encode `\'`
    # as an escaped backslash followed by a quote rather than as the apostrophe
    # the model meant. Where escapes are in play this falls through to the
    # repair path, which is where it was before.
    if "\\" not in s:
        try:
            parsed = json.loads(_NORMALIZED_QUOTED_STRINGS.transform_string(s))
            if isinstance(parsed, (dict, list)):
                return json.dumps(parsed)
        except (ValueError, ParseException):
            pass

    s_repaired_obj = repair_json(s, return_objects=True)
    if isinstance(s_repaired_obj, list):
        if len(s_repaired_obj) > 0:
            s_repaired_obj = s_repaired_obj[0]
        else:
            s_repaired_obj = None
    if s_repaired_obj is not None:
        return json.dumps(s_repaired_obj)  # type: ignore
    else:
        try:
            yaml_result = yaml.safe_load(s)
            if isinstance(yaml_result, dict):
                return json.dumps(yaml_result)
        except yaml.YAMLError:
            pass
        # If it still fails, replace any \n with space as a last resort
        s = s.replace("\n", " ")
        if is_valid_json(s):
            return s
        else:
            return None  # all failed


def extract_top_level_json(s: str) -> List[str]:
    """Extract all top-level JSON-formatted substrings from a given string.

    Args:
        s (str): The input string to search for JSON substrings.

    Returns:
        List[str]: A list of top-level JSON-formatted substrings.
    """
    # Find JSON object and array candidates
    json_candidates = get_json_candidates(s)
    maybe_repaired_jsons = map(try_repair_json_yaml, json_candidates)

    return [candidate for candidate in maybe_repaired_jsons if candidate is not None]


def top_level_json_field(s: str, f: str) -> Any:
    """
    Extract the value of a field f from a top-level JSON object.
    If there are multiple, just return the first.

    Args:
        s (str): The input string to search for JSON substrings.
        f (str): The field to extract from the JSON object.

    Returns:
        str: The value of the field f in the top-level JSON object, if any.
            Otherwise, return an empty string.

    Note:
        This function is designed to never crash. If any exception occurs during
        JSON parsing or field extraction, it gracefully returns an empty string.
    """
    try:
        jsons = extract_top_level_json(s)
        if len(jsons) == 0:
            return ""
        for j in jsons:
            try:
                json_data = json.loads(j)
                if isinstance(json_data, dict):
                    if f in json_data:
                        return json_data[f]
                elif isinstance(json_data, list):
                    # Some responses wrap candidate JSON objects in a list; scan them.
                    for item in json_data:
                        if isinstance(item, dict) and f in item:
                            return item[f]
            except (json.JSONDecodeError, TypeError, KeyError):
                # If this specific JSON fails to parse, continue to next candidate
                continue
    except Exception:
        # Catch any unexpected errors to ensure we never crash
        pass

    return ""


def datetime_to_json(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    # Let json.dumps() handle the raising of TypeError for non-serializable objects
    return obj
