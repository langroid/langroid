"""
Test Pydantic v2 subclassing behavior with OpenAICallParams.

This test demonstrates:
1. The WRONG way to subclass (without proper type annotations) - fields get dropped
2. The CORRECT way to subclass (with proper type annotations) - fields are preserved
"""

# Note: In Pydantic v2, we can't even define a class with non-annotated fields
# without getting an error. We'll use typing.ClassVar to demonstrate the issue
from typing import ClassVar

import pytest

from langroid.language_models.openai_gpt import (
    OpenAICallParams,
    OpenAIGPT,
    OpenAIGPTConfig,
)


# WRONG WAY: Using ClassVar makes them class attributes, not instance fields
class WrongCustomParams(OpenAICallParams):
    # These become class attributes, not model fields
    custom_field: ClassVar[str] = "default_value"
    another_field: ClassVar[int] = 42


# CORRECT WAY: Subclassing with proper type annotations
class CorrectCustomParams(OpenAICallParams):
    # This is the correct approach - proper type annotations
    custom_field: str = "default_value"
    another_field: int = 42


def test_wrong_way_subclass_loses_fields():
    """Test that using ClassVar makes fields class-level, not instance fields."""

    # Create an instance - note we can't pass custom fields to constructor
    # because ClassVar fields are not model fields
    wrong_params = WrongCustomParams(temperature=0.8)

    # ClassVar fields exist as class attributes only
    assert wrong_params.temperature == 0.8
    assert WrongCustomParams.custom_field == "default_value"  # Class attribute
    assert WrongCustomParams.another_field == 42  # Class attribute

    # In Pydantic v2, you can't even set ClassVar on instances - it raises an error!
    with pytest.raises(AttributeError, match="is a ClassVar"):
        wrong_params.custom_field = "test_value"

    # This is what happens in OpenAIGPT.__init__()
    copied_params = wrong_params.model_copy()

    # After model_copy(), only model fields are preserved
    assert copied_params.temperature == 0.8  # Standard field preserved

    # ClassVar fields are NOT part of the model
    dumped = copied_params.model_dump()
    assert "temperature" in dumped
    assert "custom_field" not in dumped  # Not a model field
    assert "another_field" not in dumped  # Not a model field


def test_correct_way_subclass_preserves_fields():
    """Test that subclassing with proper type annotations preserves custom fields."""

    # Create an instance with custom fields
    correct_params = CorrectCustomParams(
        temperature=0.8, custom_field="test_value", another_field=123
    )

    # Verify original params have the fields
    assert correct_params.temperature == 0.8
    assert correct_params.custom_field == "test_value"
    assert correct_params.another_field == 123

    # This is what happens in OpenAIGPT.__init__()
    copied_params = correct_params.model_copy()

    # After model_copy(), custom fields should be preserved (this is the solution)
    assert copied_params.temperature == 0.8  # Standard field preserved
    assert copied_params.custom_field == "test_value"  # Custom field preserved
    assert copied_params.another_field == 123  # Custom field preserved

    # Verify fields are in model_dump
    dumped = copied_params.model_dump()
    assert "temperature" in dumped
    assert "custom_field" in dumped
    assert "another_field" in dumped


def test_openai_gpt_preserves_custom_fields_after_fix():
    """Test that OpenAIGPT now preserves custom fields after the fix."""

    # Test with correct params
    correct_params = CorrectCustomParams(
        temperature=0.8, custom_field="integration_test", another_field=999
    )

    config = OpenAIGPTConfig(
        chat_model="gpt-3.5-turbo",  # Use a basic model for testing
        params=correct_params,
    )

    # Verify config has custom fields before OpenAIGPT.__init__()
    assert config.params.custom_field == "integration_test"
    assert config.params.another_field == 999
    assert isinstance(config.params, CorrectCustomParams)

    # This will call config.model_copy() internally
    llm = OpenAIGPT(config)

    # After the fix, params should preserve the subclass type!
    assert isinstance(llm.config.params, CorrectCustomParams)
    assert isinstance(
        llm.config.params, OpenAICallParams
    )  # Still is-a OpenAICallParams

    # Custom fields are preserved
    assert llm.config.params.custom_field == "integration_test"
    assert llm.config.params.another_field == 999

    # Test mutation independence - changes to llm.config don't affect original
    llm.config.params.custom_field = "modified"
    assert config.params.custom_field == "integration_test"  # Original unchanged


def test_workaround_set_params_after_init():
    """Test the workaround: set params after OpenAIGPT initialization."""

    # Create config with default params first
    config = OpenAIGPTConfig(chat_model="gpt-3.5-turbo")

    # Initialize OpenAIGPT
    llm = OpenAIGPT(config)

    # WORKAROUND: Set custom params after initialization
    correct_params = CorrectCustomParams(
        temperature=0.8, custom_field="workaround_test", another_field=777
    )
    llm.config.params = correct_params

    # Verify custom fields are preserved with workaround
    assert llm.config.params.custom_field == "workaround_test"
    assert llm.config.params.another_field == 777
    assert isinstance(llm.config.params, CorrectCustomParams)


def test_pydantic_v2_behavior_documentation():
    """
    Document the Pydantic v2 behavior for reference.

    In Pydantic v2:
    1. Fields without type annotations are not considered model fields
    2. They become class attributes but are not part of the model schema
    3. model_copy() only copies actual model fields (those with type annotations)
    4. This is different from Pydantic v1 where all class attributes were included

    SOLUTION: Always use proper type annotations for all fields you want to persist:
      ❌ custom_field = 'default'           # Class attribute, not model field
      ✅ custom_field: str = 'default'      # Model field, will be copied
    """
    # This is a documentation test - it always passes
    assert True
