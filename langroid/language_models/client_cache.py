"""
Client caching/singleton pattern for LLM clients to prevent connection pool exhaustion.
"""

import atexit
import hashlib
import threading
import time
import weakref
from typing import Any, Dict, Optional, Tuple, Union, cast

from cerebras.cloud.sdk import AsyncCerebras, Cerebras
from groq import AsyncGroq, Groq
from httpx import Timeout
from openai import AsyncOpenAI, OpenAI

# Cache for client instances, keyed by hashed configuration parameters.
# Value is a tuple of (client instance, last_used_monotonic_seconds).
_client_cache: Dict[str, Tuple[Any, float]] = {}
_client_cache_lock = threading.RLock()

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


def _get_cached_client(cache_key: str) -> Optional[Any]:
    """Get cached client and refresh its last-used timestamp."""
    with _client_cache_lock:
        entry = _client_cache.get(cache_key)
        if entry is None:
            return None

        client, _ = entry
        _client_cache[cache_key] = (client, time.monotonic())
        return client


def get_openai_client(
    api_key: str,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    timeout: Union[float, Timeout] = 120.0,
    default_headers: Optional[Dict[str, str]] = None,
    http_client: Optional[Any] = None,
    http_client_config: Optional[Dict[str, Any]] = None,
) -> OpenAI:
    """
    Get or create a singleton OpenAI client with the given configuration.

    Args:
        api_key: OpenAI API key
        base_url: Optional base URL for API
        organization: Optional organization ID
        timeout: Request timeout
        default_headers: Optional default headers
        http_client: Optional httpx.Client instance
        http_client_config: Optional config dict for creating httpx.Client

    Returns:
        OpenAI client instance
    """
    if isinstance(timeout, (int, float)):
        timeout = Timeout(timeout)

    # If http_client is provided directly, don't cache (complex object)
    if http_client is not None:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            timeout=timeout,
            default_headers=default_headers,
            http_client=http_client,
        )
        _all_clients.add(client)
        return client

    # If http_client_config is provided, create client from config and cache
    created_http_client = None
    if http_client_config is not None:
        try:
            from httpx import Client

            created_http_client = Client(**http_client_config)
        except ImportError:
            raise ValueError(
                "httpx is required to use http_client_config. "
                "Install it with: pip install httpx"
            )

    cache_key = _get_cache_key(
        "openai",
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        timeout=timeout,
        default_headers=default_headers,
        http_client_config=http_client_config,  # Include config in cache key
    )

    cached_client = _get_cached_client(cache_key)
    if cached_client is not None:
        return cast(OpenAI, cached_client)

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        timeout=timeout,
        default_headers=default_headers,
        http_client=created_http_client,  # Use the client created from config
    )

    with _client_cache_lock:
        _client_cache[cache_key] = (client, time.monotonic())
    _all_clients.add(client)
    return client


def get_async_openai_client(
    api_key: str,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    timeout: Union[float, Timeout] = 120.0,
    default_headers: Optional[Dict[str, str]] = None,
    http_client: Optional[Any] = None,
    http_client_config: Optional[Dict[str, Any]] = None,
) -> AsyncOpenAI:
    """
    Get or create a singleton AsyncOpenAI client with the given configuration.

    Args:
        api_key: OpenAI API key
        base_url: Optional base URL for API
        organization: Optional organization ID
        timeout: Request timeout
        default_headers: Optional default headers
        http_client: Optional httpx.AsyncClient instance
        http_client_config: Optional config dict for creating httpx.AsyncClient

    Returns:
        AsyncOpenAI client instance
    """
    if isinstance(timeout, (int, float)):
        timeout = Timeout(timeout)

    # If http_client is provided directly, don't cache (complex object)
    if http_client is not None:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            timeout=timeout,
            default_headers=default_headers,
            http_client=http_client,
        )
        _all_clients.add(client)
        return client

    # If http_client_config is provided, create async client from config and cache
    created_http_client = None
    if http_client_config is not None:
        try:
            from httpx import AsyncClient

            created_http_client = AsyncClient(**http_client_config)
        except ImportError:
            raise ValueError(
                "httpx is required to use http_client_config. "
                "Install it with: pip install httpx"
            )

    cache_key = _get_cache_key(
        "async_openai",
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        timeout=timeout,
        default_headers=default_headers,
        http_client_config=http_client_config,  # Include config in cache key
    )

    cached_client = _get_cached_client(cache_key)
    if cached_client is not None:
        return cast(AsyncOpenAI, cached_client)

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        timeout=timeout,
        default_headers=default_headers,
        http_client=created_http_client,  # Use the client created from config
    )

    with _client_cache_lock:
        _client_cache[cache_key] = (client, time.monotonic())
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

    cached_client = _get_cached_client(cache_key)
    if cached_client is not None:
        return cast(Groq, cached_client)

    client = Groq(api_key=api_key)
    with _client_cache_lock:
        _client_cache[cache_key] = (client, time.monotonic())
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

    cached_client = _get_cached_client(cache_key)
    if cached_client is not None:
        return cast(AsyncGroq, cached_client)

    client = AsyncGroq(api_key=api_key)
    with _client_cache_lock:
        _client_cache[cache_key] = (client, time.monotonic())
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

    cached_client = _get_cached_client(cache_key)
    if cached_client is not None:
        return cast(Cerebras, cached_client)

    client = Cerebras(api_key=api_key)
    with _client_cache_lock:
        _client_cache[cache_key] = (client, time.monotonic())
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

    cached_client = _get_cached_client(cache_key)
    if cached_client is not None:
        return cast(AsyncCerebras, cached_client)

    client = AsyncCerebras(api_key=api_key)
    with _client_cache_lock:
        _client_cache[cache_key] = (client, time.monotonic())
    _all_clients.add(client)
    return client


def prune_cache(max_age_seconds: float) -> int:
    """
    Clear cache entries older than the specified age.

    Args:
        max_age_seconds: Maximum age (in seconds) for cache entries to keep.
            Entries older than this value are removed.

    Returns:
        Number of cache entries removed.
    """
    if max_age_seconds < 0:
        raise ValueError("max_age_seconds must be non-negative")

    now = time.monotonic()
    with _client_cache_lock:
        stale_entries = [
            key
            for key, (_, last_used_at) in _client_cache.items()
            if now - last_used_at > max_age_seconds
        ]

        for key in stale_entries:
            _client_cache.pop(key, None)

    return len(stale_entries)


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
    with _client_cache_lock:
        _client_cache.clear()
