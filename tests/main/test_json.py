import json
import os

import pytest

from langroid.parsing.json import extract_top_level_json


@pytest.mark.parametrize(
    "s, expected",
    [
        ("nothing to see here", []),
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
    ],
)
def test_extract_top_level_json(s, expected):
    top_level_jsons = extract_top_level_json(s)
    top_level_jsons = [json.loads(s.replace("'", '"')) for s in top_level_jsons]
    expected = [json.loads(s.replace("'", '"')) for s in expected]
    assert len(top_level_jsons) == len(expected)
    assert top_level_jsons == expected


def extract_data_from_file(file_path):
    # Initialize variables
    llm_response_not_json = ""
    input_str_correct = ""
    expected_json_substrings = []
    input_str_problematic = ""
    expected_json_substrings_problematic = []

    # Open and read the file
    with open(file_path, "r") as file:
        content = file.read()

    # Split the content by '==='
    parts = content.split("===\n")

    # Extract and process each part
    for part in parts:
        if "llm_response_not_json =" in part:
            llm_response_not_json = part.split("=", 1)[1].strip()
        elif "input_str_correct =" in part:
            input_str_correct = part.split("=", 1)[1].strip()
        elif "expected_json_substrings =" in part:
            expected_json_substrings = eval(part.split("=", 1)[1].strip())
        elif "input_str_problematic =" in part:
            input_str_problematic = part.split("=", 1)[1].strip()
        elif "expected_json_substrings_problematic =" in part:
            expected_json_substrings_problematic = eval(part.split("=", 1)[1].strip())

    return (
        llm_response_not_json,
        input_str_correct,
        expected_json_substrings,
        input_str_problematic,
        expected_json_substrings_problematic,
    )


current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, "llm_msgs.txt")

(
    llm_response_not_json,
    input_str_correct,
    expected_json_substrings,
    input_str_problematic,
    expected_json_substrings_problematic,
) = extract_data_from_file(file_path)

# llm_response_not_json should be rejected by langroid before reaching the function
# extract_top_level_json because it's not in a json format


@pytest.mark.parametrize(
    "s, expected",
    [
        (
            f"{input_str_correct}",
            expected_json_substrings,
        ),
        (
            f"{input_str_problematic}",
            expected_json_substrings_problematic,
        ),
    ],
)
def test_extract_top_level_json_LLM_response(s, expected):
    top_level_jsons = extract_top_level_json(s)
    top_level_jsons = [json.loads(s.replace("'", '"')) for s in top_level_jsons]
    assert len(top_level_jsons) == len(expected)
    assert top_level_jsons == expected
