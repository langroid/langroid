"""
Compatibility layer for Pydantic v2 migration.

This module now imports directly from Pydantic v2 since all internal code
has been migrated to use Pydantic v2 patterns.
"""

# Import everything from pydantic v2
from langroid.pydantic_v1 import *  # noqa: F403, F401

# Import BaseSettings from pydantic-settings v2
from pydantic_settings import BaseSettings  # noqa: F401

# Explicitly re-export commonly used items for better IDE support and type checking
from langroid.pydantic_v1 import (  # noqa: F401
    BaseModel,
    Field,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
    create_model,
    HttpUrl,
    AnyUrl,
    parse_obj_as,
)

# Legacy names that map to v2 equivalents
validator = field_validator  # noqa: F401
root_validator = model_validator  # noqa: F401
