"""
Compatibility layer for Pydantic v2 migration.

This module now imports directly from Pydantic v2 since all internal code
has been migrated to use Pydantic v2 patterns.
"""

# Import from pydantic.main but don't trigger the warning again
# The warning is already shown when importing from langroid.pydantic_v1
from pydantic.main import *  # noqa: F403, F401
