"""
Client caching/singleton pattern for LLM clients to prevent connection pool exhaustion.
"""

import atexit
import hashlib
import weakref
from typing import Any, Dict, Optional, Union, cast

from cerebras.cloud.sdk import AsyncCerebras, Cerebras
from groq import AsyncGroq, Groq
from httpx import Timeout
from openai import AsyncOpenAI, OpenAI

# Cache for client instances, keyed by hashed configuration parameters
_client_cache: Dict[str, Any] = {}

# Keep track of clients for cleanup
_all_clients: weakref.WeakSet[Any] = weakref.WeakSet()


def _get_cache_key(client_type: str, **kwargs: Any) -> str:
    """
    Generate a cache key from client type and configuration parameters.
    Uses the same approach as OpenAIGPT._cache_lookup for consistency.

    Args:
        client_type: Type of client (e.g., "openai", "groq", "cerebras")
        **kwargs: Configuration parameters (api_key, base_url, timeout, etc.)

    Returns:
        SHA256 hash of the configuration as a hex string
    """
    # Convert kwargs to sorted string representation
    sorted_kwargs_str = str(sorted(kwargs.items()))

    # Create raw key combining client type and sorted kwargs
    raw_key = f"{client_type}:{sorted_kwargs_str}"

    # Hash the key for consistent length and to handle complex objects
    hashed_key = hashlib.sha256(raw_key.encode()).hexdigest()

    return hashed_key


def get_openai_client(
    api_key: str,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    timeout: Union[float, Timeout] = 120.0,
    default_headers: Optional[Dict[str, str]] = None,
) -> OpenAI:
    """
    Get or create a singleton OpenAI client with the given configuration.

    Args:
        api_key: OpenAI API key
        base_url: Optional base URL for API
        organization: Optional organization ID
        timeout: Request timeout
        default_headers: Optional default headers

    Returns:
        OpenAI client instance
    """
    if isinstance(timeout, (int, float)):
        timeout = Timeout(timeout)

    cache_key = _get_cache_key(
        "openai",
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        timeout=timeout,
        default_headers=default_headers,
    )

    if cache_key in _client_cache:
        return cast(OpenAI, _client_cache[cache_key])

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        timeout=timeout,
        default_headers=default_headers,
    )

    _client_cache[cache_key] = client
    _all_clients.add(client)
    return client


def get_async_openai_client(
    api_key: str,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    timeout: Union[float, Timeout] = 120.0,
    default_headers: Optional[Dict[str, str]] = None,
) -> AsyncOpenAI:
    """
    Get or create a singleton AsyncOpenAI client with the given configuration.

    Args:
        api_key: OpenAI API key
        base_url: Optional base URL for API
        organization: Optional organization ID
        timeout: Request timeout
        default_headers: Optional default headers

    Returns:
        AsyncOpenAI client instance
    """
    if isinstance(timeout, (int, float)):
        timeout = Timeout(timeout)

    cache_key = _get_cache_key(
        "async_openai",
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        timeout=timeout,
        default_headers=default_headers,
    )

    if cache_key in _client_cache:
        return cast(AsyncOpenAI, _client_cache[cache_key])

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        timeout=timeout,
        default_headers=default_headers,
    )

    _client_cache[cache_key] = client
    _all_clients.add(client)
    return client


def get_groq_client(api_key: str) -> Groq:
    """
    Get or create a singleton Groq client with the given configuration.

    Args:
        api_key: Groq API key

    Returns:
        Groq client instance
    """
    cache_key = _get_cache_key("groq", api_key=api_key)

    if cache_key in _client_cache:
        return cast(Groq, _client_cache[cache_key])

    client = Groq(api_key=api_key)
    _client_cache[cache_key] = client
    _all_clients.add(client)
    return client


def get_async_groq_client(api_key: str) -> AsyncGroq:
    """
    Get or create a singleton AsyncGroq client with the given configuration.

    Args:
        api_key: Groq API key

    Returns:
        AsyncGroq client instance
    """
    cache_key = _get_cache_key("async_groq", api_key=api_key)

    if cache_key in _client_cache:
        return cast(AsyncGroq, _client_cache[cache_key])

    client = AsyncGroq(api_key=api_key)
    _client_cache[cache_key] = client
    _all_clients.add(client)
    return client


def get_cerebras_client(api_key: str) -> Cerebras:
    """
    Get or create a singleton Cerebras client with the given configuration.

    Args:
        api_key: Cerebras API key

    Returns:
        Cerebras client instance
    """
    cache_key = _get_cache_key("cerebras", api_key=api_key)

    if cache_key in _client_cache:
        return cast(Cerebras, _client_cache[cache_key])

    client = Cerebras(api_key=api_key)
    _client_cache[cache_key] = client
    _all_clients.add(client)
    return client


def get_async_cerebras_client(api_key: str) -> AsyncCerebras:
    """
    Get or create a singleton AsyncCerebras client with the given configuration.

    Args:
        api_key: Cerebras API key

    Returns:
        AsyncCerebras client instance
    """
    cache_key = _get_cache_key("async_cerebras", api_key=api_key)

    if cache_key in _client_cache:
        return cast(AsyncCerebras, _client_cache[cache_key])

    client = AsyncCerebras(api_key=api_key)
    _client_cache[cache_key] = client
    _all_clients.add(client)
    return client


def _cleanup_clients() -> None:
    """
    Cleanup function to close all cached clients on exit.
    Called automatically via atexit.
    """
    import inspect

    for client in list(_all_clients):
        if hasattr(client, "close") and callable(client.close):
            try:
                # Check if close is a coroutine function (async)
                if inspect.iscoroutinefunction(client.close):
                    # For async clients, we can't await in atexit
                    # They will be cleaned up by the OS
                    pass
                else:
                    # Sync clients can be closed directly
                    client.close()
            except Exception:
                pass  # Ignore errors during cleanup


# Register cleanup function to run on exit
atexit.register(_cleanup_clients)


# For testing purposes
def _clear_cache() -> None:
    """Clear the client cache. Only for testing."""
    _client_cache.clear()
