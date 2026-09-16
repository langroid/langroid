"""
Compatibility shim over the ``httpx`` / ``httpx2`` split in the OpenAI SDK.

``openai < 3`` is built on ``httpx``; ``openai >= 3`` replaced it with
``httpx2`` (the Pydantic-maintained successor). Both expose the same
``Client`` / ``AsyncClient`` / ``Timeout`` API, but each SDK major only
depends on its own family, so langroid must import the family that matches
the installed ``openai`` major rather than hard-coding ``httpx``.

Everything in langroid that needs an HTTP-client type should go through this
module instead of importing ``httpx`` or ``httpx2`` directly.
"""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

import openai

if TYPE_CHECKING:
    # mypy runs against openai 2.x (httpx); the runtime binding below picks
    # the right family for whichever SDK major is installed.
    from httpx import Timeout


def _openai_major() -> int:
    """Return the major version of the installed ``openai`` SDK."""
    return int(openai.__version__.split(".")[0])


HTTPX_MODULE_NAME: str = "httpx2" if _openai_major() >= 3 else "httpx"
"""Name of the httpx family the installed ``openai`` SDK is built on."""


def import_httpx() -> ModuleType:
    """Import and return the httpx family matching the installed openai SDK.

    The import is performed on every call (cheap: ``sys.modules`` caches it)
    so callers can wrap it in ``try/except ImportError`` and report a
    family-specific install hint.

    Returns:
        The ``httpx`` module (openai 2.x) or the ``httpx2`` module
        (openai 3.x).

    Raises:
        ImportError: If the required package is not installed.
    """
    return import_module(HTTPX_MODULE_NAME)


def missing_httpx_message() -> str:
    """Error text for ``http_client_config`` when the httpx family is missing."""
    return (
        f"{HTTPX_MODULE_NAME} is required to use http_client_config. "
        f"Install it with: pip install {HTTPX_MODULE_NAME}"
    )


if not TYPE_CHECKING:
    Timeout = import_httpx().Timeout

__all__ = ["HTTPX_MODULE_NAME", "Timeout", "import_httpx", "missing_httpx_message"]
