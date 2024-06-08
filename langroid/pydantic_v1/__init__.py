"""
If we're on Pydantic v2, use the v1 namespace, else just use the main namespace.

This allows compatibility with both Pydantic v1 and v2
"""

try:
    from pydantic.v1 import *  # noqa: F403, F401
except ImportError:
    from pydantic import *  # type: ignore # noqa: F403, F401
