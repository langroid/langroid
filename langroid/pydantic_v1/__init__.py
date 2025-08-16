"""
Compatibility layer for Langroid's Pydantic migration.

IMPORTANT: You are importing from langroid.pydantic_v1 but getting Pydantic v2 classes!
Langroid has fully migrated to Pydantic v2, and this compatibility layer is deprecated.
"""

import warnings
import logging

logger = logging.getLogger(__name__)

# Only show the visual warning, not the standard deprecation warning
# The standard warning is too noisy and shows the import line
logger.warning(
    """
╔════════════════════════════════════════════════════════════════════════╗
║                    ⚠️  DEPRECATION WARNING ⚠️                          ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  You are importing from langroid.pydantic_v1, but you're actually      ║
║  getting Pydantic v2 classes. Langroid has fully migrated to v2.       ║
║                                                                        ║
║  Please update your imports:                                           ║
║    OLD: from langroid.pydantic_v1 import BaseModel, Field              ║
║    NEW: from pydantic import BaseModel, Field                          ║
║                                                                        ║
║  Also ensure your code uses Pydantic v2 patterns:                      ║
║    • Use model_dump() instead of dict()                                ║
║    • Use model_dump_json() instead of json()                           ║
║    • Use ConfigDict instead of class Config                            ║
║    • Use model_validate() instead of parse_obj()                       ║
║                                                                        ║
║  This compatibility layer will be removed in a future version.         ║
╚════════════════════════════════════════════════════════════════════════╝
"""
)

# Import from pydantic v2 directly (not from pydantic.v1)
# This allows existing code to continue working if it's already v2-compatible
from pydantic import *  # noqa: F403, F401

# BaseSettings has moved in v2, import it explicitly
try:
    from pydantic_settings import BaseSettings  # noqa: F401
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings  # type: ignore[no-redef] # noqa: F401

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
