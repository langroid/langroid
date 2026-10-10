import ast
import json
from datetime import datetime
from typing import Any, Dict, Iterator, List, Union

import yaml
from json_repair import repair_json
from pyparsing import (
    FollowedBy,
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
        if start < end - line_start < stop and line[:start].rstrip().endswith(
            (":", ",", "[")
        ):
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
# truncated mid-string, say) it costs a ~2.6x constant factor over the
# single-line grammar. That factor is unavoidable within this approach, so the
# length bound is what keeps the absolute cost small: 8 KiB caps the extra work
# at a few hundred ms, where 64 KiB allowed ~3s on a 63 KB truncated reply.
# Well-formed input of any size is unaffected — it closes immediately and never
# enters the slow scan. Multiline recovery is therefore best-effort, and applies
# to tool calls up to 8 KiB; larger ones keep the original (truncating)
# behavior rather than stalling extraction.
_MAX_MULTILINE_INPUT_CHARS = 8 * 1024
_MAX_MULTILINE_QUOTES = 512


def get_json_candidates(s: str) -> List[str]:
    """Get top-level JSON candidates, i.e. strings between curly braces."""
    curly_braces = (
        _CURLY_BRACES
        if len(s) <= _MAX_MULTILINE_INPUT_CHARS
        and s.count('"') + s.count("'") <= _MAX_MULTILINE_QUOTES
        else _SINGLE_LINE_CURLY_BRACES
    )

    # Parse the string
    try:
        results = curly_braces.search_string(s)
        # Properly convert nested lists to strings
        return [r[0] for r in results]
    except Exception:
        return []


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
