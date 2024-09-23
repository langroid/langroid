import ast
import json
from datetime import datetime
from typing import Any, Dict, Iterator, List, Union

import yaml
from json_repair import repair_json
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
