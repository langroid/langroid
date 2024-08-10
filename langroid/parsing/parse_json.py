import ast
import json
from datetime import datetime
from typing import Any, Dict, Iterator, List, Union

import yaml
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


def add_quotes(s: str) -> str:
    """
    Replace accidentally un-quoted string-like keys and values in a potential json str.
    Intended to handle cases where a weak LLM may produce a JSON-like string
    containing, e.g. "rent": DO-NOT-KNOW, where it "forgot" to put quotes on the value,
    or city: "New York" where it "forgot" to put quotes on the key.
    It will even handle cases like 'address: do not know'.

    Got this fiendishly clever solution from
    https://stackoverflow.com/a/66053900/10940584
    Far better/safer than trying to do it with regexes.

    Args:
    - s (str): The potential JSON string to parse.

    Returns:
    - str: The (potential) JSON string with un-quoted string-like values
        replaced by quoted values.
    """
    if is_valid_json(s):
        return s
    try:
        dct = yaml.load(s, yaml.SafeLoader)
        return json.dumps(dct)
    except Exception:
        return s


def parse_imperfect_json(json_string: str) -> Union[Dict[str, Any], List[Any]]:
    if not json_string.strip():
        raise ValueError("Empty string is not valid JSON")

    # First, try parsing with ast.literal_eval
    try:
        result = ast.literal_eval(json_string)
        if isinstance(result, (dict, list)):
            return result
    except (ValueError, SyntaxError):
        pass

    # If ast.literal_eval fails or returns non-dict/list, try json.loads
    try:
        json_string = add_quotes(json_string)
        result = json.loads(json_string)
        if isinstance(result, (dict, list)):
            return result
    except json.JSONDecodeError:
        try:
            # fallback on yaml
            yaml_result = yaml.safe_load(json_string)
            if isinstance(yaml_result, (dict, list)):
                return yaml_result
        except yaml.YAMLError:
            pass

    try:
        # last resort: try to repair the json using a lib
        from json_repair import repair_json

        repaired_json = repair_json(json_string)
        result = json.loads(repaired_json)
        if isinstance(result, (dict, list)):
            return result
    except Exception:
        pass

    # If all methods fail, raise ValueError
    raise ValueError(f"Unable to parse as JSON: {json_string}")


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
    candidates = [add_quotes(candidate) for candidate in normalized_candidates]
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


def datetime_to_json(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    # Let json.dumps() handle the raising of TypeError for non-serializable objects
    return obj
