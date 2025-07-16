import pytest

from langroid.pydantic_v1 import BaseModel, ConfigDict
from langroid.utils.pydantic_utils import extract_fields, flatten_dict


class DetailsModel(BaseModel):
    height: float
    weight: float


class TestModel(BaseModel):
    name: str
    age: int
    details: DetailsModel

    model_config = ConfigDict(populate_by_name=True)


def test_extract_fields():
    # Create an instance of TestModel with nested DetailsModel
    test_instance = TestModel(
        name="John Doe", age=30, details=DetailsModel(height=180.5, weight=75.0)
    )

    # Test with single field
    result = extract_fields(test_instance, ["name"])
    assert result == {"name": "John Doe"}

    # Test with multiple fields
    result = extract_fields(test_instance, ["name", "age", "weight"])
    assert result == {"name": "John Doe", "age": 30, "weight": 75.0}

    # Test with nested field using dot notation
    # Note we only retain the LAST part of the field name
    result = extract_fields(test_instance, ["details.height"])
    assert result == {"height": 180.5}

    # Test with nested field using non-dot notation
    result = extract_fields(test_instance, ["weight"])
    assert result == {"weight": 75.0}

    # Test with non-existent field
    result = extract_fields(test_instance, ["non_existent_field"])
    assert result == {}

    # Test with empty fields list
    result = extract_fields(test_instance, [])
    assert result == {}


@pytest.mark.parametrize(
    "input_dict, expected_output",
    [
        ({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2, "c": 3}),
        ({"a": 1, "b": {"c": 2, "d": 3}, "e": 4}, {"a": 1, "b.c": 2, "b.d": 3, "e": 4}),
        ({"a": {"b": {"c": {"d": 1}}}}, {"a.b.c.d": 1}),
        ({"a": [1, 2, 3], "b": {"c": [4, 5, 6]}}, {"a": [1, 2, 3], "b.c": [4, 5, 6]}),
        ({"a": 1, "b": {}, "c": 3}, {"a": 1, "c": 3}),
        ({}, {}),
        ({"a": None, "b": {"c": None}}, {"a": None, "b.c": None}),
    ],
)
def test_flatten_dict(input_dict, expected_output):
    assert flatten_dict(input_dict) == expected_output


@pytest.mark.parametrize(
    "input_dict, separator, expected_output",
    [
        ({"a": 1, "b": {"c": 2, "d": 3}}, "__", {"a": 1, "b__c": 2, "b__d": 3}),
        ({"x": {"y": {"z": 1}}}, "->", {"x->y->z": 1}),
    ],
)
def test_flatten_dict_custom_separator(input_dict, separator, expected_output):
    assert flatten_dict(input_dict, sep=separator) == expected_output
