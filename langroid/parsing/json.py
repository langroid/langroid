import json
from typing import List

import regex


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


def extract_top_level_json(s: str) -> List[str]:
    """Extract all top-level JSON-formatted substrings from a given string.

    Args:
        s (str): The input string to search for JSON substrings.

    Returns:
        List[str]: A list of top-level JSON-formatted substrings.
    """
    # Find JSON object and array candidates using regular expressions
    json_candidates = regex.findall(r"(?<!\\)(?:\\\\)*\{(?:[^{}]|(?R))*\}", s)

    top_level_jsons = [
        candidate for candidate in json_candidates if is_valid_json(candidate)
    ]

    return top_level_jsons
