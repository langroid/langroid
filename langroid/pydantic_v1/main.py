"""
Compatibility layer for Pydantic v2 migration.

This module now imports directly from Pydantic v2 since all internal code
has been migrated to use Pydantic v2 patterns.
"""

# Explicitly export BaseModel for better type checking
from pydantic.main import *  # noqa: F403, F401

from langroid.pydantic_v1 import BaseModel  # noqa: F401
