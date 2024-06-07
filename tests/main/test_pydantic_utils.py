from pydantic.v1 import BaseModel

from langroid.utils.pydantic_utils import extract_fields


class DetailsModel(BaseModel):
    height: float
    weight: float


class TestModel(BaseModel):
    name: str
    age: int
    details: DetailsModel

    class Config:
        allow_population_by_field_name = True


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
