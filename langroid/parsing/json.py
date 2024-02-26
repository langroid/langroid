import json
import re
from typing import Any, Iterator, List

from pyparsing import nestedExpr, originalTextFor


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


def get_json_candidates(s: str) -> List[str]:
    """Get top-level JSON candidates, i.e. strings between curly braces."""
    # Define the grammar for matching curly braces
    curly_braces = originalTextFor(nestedExpr("{", "}"))

    # Parse the string
    try:
        results = curly_braces.searchString(s)
        # Properly convert nested lists to strings
        return [r[0] for r in results]
    except Exception:
        return []


def replace_undefined(s: str, undefined_placeholder: str = '"<undefined>"') -> str:
    """
    Replace undefined values in a potential json str with a placeholder.

    Args:
    - s (str): The potential JSON string to parse.
    - undefined_placeholder (str): The placeholder or error message
        for undefined values.

    Returns:
    - str: The (potential) JSON string with undefined values
        replaced by the placeholder.
    """

    # Preprocess the string to replace undefined values with the placeholder
    # This regex looks for patterns like ": <identifier>" and replaces them
    # with the placeholder.
    # It's a simple approach and might need adjustments for complex cases
    # This is an attempt to handle cases where a weak LLM may produce
    # a JSON-like string without quotes around some values, e.g.
    # {"rent": DO-NOT-KNOW }
    preprocessed_s = re.sub(
        r":\s*([a-zA-Z_][a-zA-Z_0-9\-]*)", f": {undefined_placeholder}", s
    )

    # Now, attempt to parse the preprocessed string as JSON
    try:
        return preprocessed_s
    except Exception:
        # If parsing fails, return an error message instead
        # (this should be rare after preprocessing)
        return s


def repair_newlines(s: str) -> str:
    """
    Attempt to load as json, and if it fails, try with newlines replaced by space.
    Intended to handle cases where weak LLMs produce JSON-like strings where
    some string-values contain explicit newlines, e.g.:
    {"text": "This is a text\n with a newline"}
    These would not be valid JSON, so we try to clean them up here.
    """
    try:
        json.loads(s)
        return s
    except Exception:
        try:
            s = s.replace("\n", " ")
            json.loads(s)
            return s
        except Exception:
            return s


def extract_top_level_json(s: str) -> List[str]:
    """Extract all top-level JSON-formatted substrings from a given string.

    Args:
        s (str): The input string to search for JSON substrings.

    Returns:
        List[str]: A list of top-level JSON-formatted substrings.
    """
    # Find JSON object and array candidates
    json_candidates = get_json_candidates(s)

    normalized_candidates = [
        candidate.replace("\\{", "{").replace("\\}", "}").replace("\\_", "_")
        for candidate in json_candidates
    ]
    candidates = [replace_undefined(candidate) for candidate in normalized_candidates]
    candidates = [repair_newlines(candidate) for candidate in candidates]
    top_level_jsons = [
        candidate for candidate in candidates if is_valid_json(candidate)
    ]

    return top_level_jsons


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
    """

    jsons = extract_top_level_json(s)
    if len(jsons) == 0:
        return ""
    for j in jsons:
        json_data = json.loads(j)
        if f in json_data:
            return json_data[f]

    return ""
