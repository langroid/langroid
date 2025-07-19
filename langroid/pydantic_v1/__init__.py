"""
Compatibility layer for Pydantic v2 migration.

This module now imports directly from Pydantic v2 since all internal code
has been migrated to use Pydantic v2 patterns.
"""

# Import everything from pydantic v2
from pydantic import *  # noqa: F403, F401

# Import BaseSettings and SettingsConfigDict from pydantic-settings v2
from pydantic_settings import BaseSettings, SettingsConfigDict  # noqa: F401

# Explicitly re-export commonly used items for better IDE support and type checking
from pydantic import (  # noqa: F401
    BaseModel,
    Field,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
    create_model,
    HttpUrl,
    AnyUrl,
    TypeAdapter,
    parse_obj_as,
)

# Legacy names are already provided by pydantic v2 for backward compatibility
# No need to redefine validator and root_validator as they are already imported above

# Explicitly export all items for mypy
__all__ = [
    "BaseModel",
    "BaseSettings",
    "SettingsConfigDict",
    "Field",
    "ConfigDict",
    "ValidationError",
    "field_validator",
    "model_validator",
    "create_model",
    "HttpUrl",
    "AnyUrl",
    "TypeAdapter",
    "parse_obj_as",
    "validator",
    "root_validator",
]
