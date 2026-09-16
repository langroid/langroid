"""
Tests for langroid.language_models.httpx_compat (GitHub issue #1138).

openai < 3 is built on ``httpx``; openai >= 3 is built on ``httpx2``. The
shim must pick the family the installed SDK actually uses, so that the
``Timeout`` / ``Client`` / ``AsyncClient`` objects langroid constructs are
accepted by the SDK's client constructors.
"""

import sys

import openai
import pytest
from openai import AsyncOpenAI, OpenAI

from langroid.language_models import httpx_compat
from langroid.language_models.httpx_compat import (
    HTTPX_MODULE_NAME,
    Timeout,
    import_httpx,
    missing_httpx_message,
)


def test_module_name_matches_openai_major():
    major = int(openai.__version__.split(".")[0])
    assert HTTPX_MODULE_NAME == ("httpx2" if major >= 3 else "httpx")


def test_import_httpx_returns_sdk_family():
    """The shim's family is the one the installed SDK re-exports."""
    mod = import_httpx()
    assert mod.__name__ == HTTPX_MODULE_NAME
    # openai re-exports its own family's Timeout in both 2.x and 3.x.
    assert mod.Timeout is openai.Timeout
    assert Timeout is openai.Timeout


def test_import_httpx_raises_import_error_when_family_missing():
    real = sys.modules[HTTPX_MODULE_NAME]
    sys.modules[HTTPX_MODULE_NAME] = None  # type: ignore[assignment]
    try:
        with pytest.raises(ImportError):
            import_httpx()
    finally:
        sys.modules[HTTPX_MODULE_NAME] = real
    # importable again once the module is restored
    assert import_httpx() is real


def test_missing_httpx_message_names_family():
    msg = missing_httpx_message()
    assert msg == (
        f"{HTTPX_MODULE_NAME} is required to use http_client_config. "
        f"Install it with: pip install {HTTPX_MODULE_NAME}"
    )


def test_sdk_accepts_shim_clients_and_timeout():
    """Objects built from the shim's family pass the SDK's type checks."""
    mod = import_httpx()
    sync_client = OpenAI(
        api_key="test-key",
        timeout=Timeout(7.0),
        http_client=mod.Client(),
    )
    async_client = AsyncOpenAI(
        api_key="test-key",
        timeout=Timeout(7.0),
        http_client=mod.AsyncClient(),
    )
    try:
        assert sync_client.timeout == Timeout(7.0)
        assert async_client.timeout == Timeout(7.0)
    finally:
        sync_client.close()


def test_shim_exports():
    assert set(httpx_compat.__all__) == {
        "HTTPX_MODULE_NAME",
        "Timeout",
        "import_httpx",
        "missing_httpx_message",
    }
