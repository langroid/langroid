"""
Tests for OpenAIGPT client caching functionality.
"""

import sys
import threading
import time
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI

import langroid.language_models.client_cache as client_cache_module
from langroid.language_models.client_cache import (
    _clear_cache,
    get_async_openai_client,
    get_cerebras_client,
    get_groq_client,
    get_openai_client,
    prune_cache,
)
from langroid.language_models.openai_gpt import OpenAIGPT, OpenAIGPTConfig


class TestOpenAIGPTClientCache:
    """Test client caching functionality for OpenAIGPT."""

    def setup_method(self):
        """Clear cache before each test."""
        _clear_cache()

    def test_openai_client_singleton(self):
        """Test that same config returns same OpenAI client instance."""
        api_key = "test-key-123"
        base_url = "https://api.test.com"

        # Get client twice with same config
        client1 = get_openai_client(api_key=api_key, base_url=base_url)
        client2 = get_openai_client(api_key=api_key, base_url=base_url)

        # Should be same instance
        assert client1 is client2

    def test_openai_client_different_config(self):
        """Test that different configs return different OpenAI client instances."""
        # Different API keys should result in different clients
        client1 = get_openai_client(api_key="key1")
        client2 = get_openai_client(api_key="key2")
        assert client1 is not client2

    def test_async_openai_client_singleton(self):
        """Test that same config returns same AsyncOpenAI client instance."""
        api_key = "test-key-async"

        client1 = get_async_openai_client(api_key=api_key)
        client2 = get_async_openai_client(api_key=api_key)

        assert client1 is client2

    def test_groq_client_singleton(self):
        """Test that same config returns same Groq client instance."""
        api_key = "groq-test-key"

        client1 = get_groq_client(api_key=api_key)
        client2 = get_groq_client(api_key=api_key)

        assert client1 is client2

    def test_prune_cache_removes_stale_entries(self, monkeypatch):
        """Test eviction of cache entries older than the specified max age."""
        fake_now = [1000.0]

        monkeypatch.setattr(client_cache_module.time, "monotonic", lambda: fake_now[0])

        client1 = get_openai_client(api_key="test-key-stale")
        fake_now[0] += 20.0

        removed = prune_cache(5.0)

        assert removed == 1

        client2 = get_openai_client(api_key="test-key-stale")
        assert client1 is not client2

    def test_prune_cache_keeps_fresh_entries(self, monkeypatch):
        """Test fresh cache entries are retained when below max age."""
        fake_now = [2000.0]

        monkeypatch.setattr(client_cache_module.time, "monotonic", lambda: fake_now[0])

        client1 = get_openai_client(api_key="test-key-fresh")
        fake_now[0] += 1.0

        removed = prune_cache(5.0)

        assert removed == 0

        client2 = get_openai_client(api_key="test-key-fresh")
        assert client1 is client2

    def test_cache_age_refreshes_on_use(self, monkeypatch):
        """Test cache entry last-used timestamp is refreshed on cache hits."""
        fake_now = [3000.0]

        monkeypatch.setattr(client_cache_module.time, "monotonic", lambda: fake_now[0])

        client1 = get_openai_client(api_key="test-key-refresh")

        # Use client again from cache; this should refresh last-used time.
        fake_now[0] += 3.0
        client2 = get_openai_client(api_key="test-key-refresh")
        assert client1 is client2

        # At this point age since last use is only 3s, so it should not be evicted.
        fake_now[0] += 3.0
        removed = prune_cache(5.0)
        assert removed == 0

        client3 = get_openai_client(api_key="test-key-refresh")
        assert client3 is client1

    def test_openai_client_cache_hit_does_not_create_extra_httpx_client(
        self, monkeypatch
    ):
        """Test cache hits avoid allocating a new sync transport for config."""
        created_clients: list[httpx.Client] = []
        real_client = httpx.Client

        class TrackingClient(real_client):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created_clients.append(self)

        monkeypatch.setattr(httpx, "Client", TrackingClient)

        client1 = get_openai_client(
            api_key="test-key-http-client-config",
            http_client_config={"timeout": 1.0},
        )
        client2 = get_openai_client(
            api_key="test-key-http-client-config",
            http_client_config={"timeout": 1.0},
        )

        assert client1 is client2
        assert len(created_clients) == 1

    def test_async_openai_client_cache_hit_does_not_create_extra_httpx_client(
        self, monkeypatch
    ):
        """Test cache hits avoid allocating a new async transport for config."""
        created_clients: list[httpx.AsyncClient] = []
        real_async_client = httpx.AsyncClient

        class TrackingAsyncClient(real_async_client):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created_clients.append(self)

        monkeypatch.setattr(httpx, "AsyncClient", TrackingAsyncClient)

        client1 = get_async_openai_client(
            api_key="test-key-async-http-client-config",
            http_client_config={"timeout": 1.0},
        )
        client2 = get_async_openai_client(
            api_key="test-key-async-http-client-config",
            http_client_config={"timeout": 1.0},
        )

        assert client1 is client2
        assert len(created_clients) == 1

    @pytest.mark.parametrize(
        "getter,httpx_attr,openai_cls,api_key",
        [
            (get_openai_client, "Client", OpenAI, "test-key-race-sync"),
            (
                get_async_openai_client,
                "AsyncClient",
                AsyncOpenAI,
                "test-key-race-async",
            ),
        ],
        ids=["sync", "async"],
    )
    def test_concurrent_misses_construct_httpx_client_once(
        self, monkeypatch, getter, httpx_attr, openai_cls, api_key
    ):
        """Racing identical misses build exactly one transport per entry.

        Miss construction is serialized under the cache lock, so threads
        racing on the same argument tuple must share a single client and
        construct at most one httpx transport.
        """
        created_http_clients: list[Any] = []
        real_http_client_cls = getattr(httpx, httpx_attr)
        n_threads = 8
        barrier = threading.Barrier(n_threads)

        class BlockingTrackingHttpClient(real_http_client_cls):
            def __init__(self, *args, **kwargs):
                created_http_clients.append(self)
                # Stay inside the constructor briefly so unserialized
                # construction by racing threads would be detected.
                time.sleep(0.15)
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(httpx, httpx_attr, BlockingTrackingHttpClient)

        results: list[Any] = []
        errors: list[BaseException] = []

        def get_client():
            try:
                barrier.wait(timeout=10)
                results.append(
                    getter(
                        api_key=api_key,
                        http_client_config={"timeout": 1.0},
                    )
                )
            except BaseException as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=get_client, daemon=True) for _ in range(n_threads)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert all(not thread.is_alive() for thread in threads)
        assert errors == []
        assert len(results) == n_threads
        assert all(client is results[0] for client in results)
        assert isinstance(results[0], openai_cls)
        assert len(created_http_clients) == 1

    @pytest.mark.parametrize(
        "getter,httpx_attr,openai_cls,api_key",
        [
            (get_openai_client, "Client", OpenAI, "test-key-flaky-sync"),
            (
                get_async_openai_client,
                "AsyncClient",
                AsyncOpenAI,
                "test-key-flaky-async",
            ),
        ],
        ids=["sync", "async"],
    )
    def test_httpx_construction_failure_propagates_and_is_not_cached(
        self, monkeypatch, getter, httpx_attr, openai_cls, api_key
    ):
        """Failed transport construction propagates and caches nothing.

        The construction error must reach the caller, the cache lock must
        be released, and no failed/partial client may be cached: a retry
        with identical arguments constructs a fresh client.
        """
        construction_calls: list[Any] = []
        real_http_client_cls = getattr(httpx, httpx_attr)

        class FlakyHttpClient(real_http_client_cls):
            def __init__(self, *args, **kwargs):
                construction_calls.append(self)
                if len(construction_calls) == 1:
                    raise RuntimeError("transport construction failed")
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(httpx, httpx_attr, FlakyHttpClient)

        def call_getter():
            return getter(
                api_key=api_key,
                http_client_config={"timeout": 1.0},
            )

        with pytest.raises(RuntimeError, match="transport construction failed"):
            call_getter()
        assert len(construction_calls) == 1

        # Retry on another thread: it must not deadlock on a leaked cache
        # lock (an RLock re-acquired from this thread would mask a leak),
        # and it must not observe a cached failed/partial client.
        retry_results: list[Any] = []
        retry_errors: list[BaseException] = []

        def retry():
            try:
                retry_results.append(call_getter())
            except BaseException as exc:
                retry_errors.append(exc)

        thread = threading.Thread(target=retry, daemon=True)
        thread.start()
        thread.join(timeout=10)

        assert not thread.is_alive(), "cache lock was not released"
        assert retry_errors == []
        assert len(retry_results) == 1
        client = retry_results[0]
        assert isinstance(client, openai_cls)
        assert len(construction_calls) == 2

        # A further identical call is a plain cache hit: same client,
        # and no additional transport construction.
        assert call_getter() is client
        assert len(construction_calls) == 2

    @pytest.mark.parametrize(
        "getter,httpx_attr,api_key",
        [
            (get_openai_client, "Client", "test-key-import-error-sync"),
            (
                get_async_openai_client,
                "AsyncClient",
                "test-key-import-error-async",
            ),
        ],
        ids=["sync", "async"],
    )
    def test_httpx_constructor_import_error_propagates_and_is_not_cached(
        self, monkeypatch, getter, httpx_attr, api_key
    ):
        """Constructor ImportError propagates unchanged and caches nothing."""
        constructor_error = ImportError("optional transport dependency is missing")
        real_http_client_cls = getattr(httpx, httpx_attr)

        class FailingHttpClient(real_http_client_cls):
            def __init__(self, *args, **kwargs):
                raise constructor_error

        monkeypatch.setattr(httpx, httpx_attr, FailingHttpClient)

        with pytest.raises(ImportError) as excinfo:
            getter(
                api_key=api_key,
                http_client_config={"timeout": 1.0},
            )

        assert excinfo.value is constructor_error
        assert len(client_cache_module._client_cache) == 0

    @pytest.mark.parametrize(
        "getter,openai_cls,api_key",
        [
            (get_openai_client, OpenAI, "test-key-no-httpx-sync"),
            (
                get_async_openai_client,
                AsyncOpenAI,
                "test-key-no-httpx-async",
            ),
        ],
        ids=["sync", "async"],
    )
    def test_missing_httpx_raises_value_error_and_is_not_cached(
        self, getter, openai_cls, api_key
    ):
        """Unimportable httpx plus http_client_config raises ValueError.

        When ``from httpx import Client`` / ``from httpx import
        AsyncClient`` fails with ImportError, the getter must raise the
        documented ValueError, cache nothing, and construct/cache
        normally once httpx is importable again.
        """

        def call_getter():
            return getter(
                api_key=api_key,
                http_client_config={"timeout": 1.0},
            )

        real_httpx = sys.modules["httpx"]
        # Simulate an environment without httpx: a None entry in
        # sys.modules makes ``from httpx import ...`` raise ImportError.
        sys.modules["httpx"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ValueError) as excinfo:
                call_getter()
        finally:
            sys.modules["httpx"] = real_httpx

        assert str(excinfo.value) == (
            "httpx is required to use http_client_config. "
            "Install it with: pip install httpx"
        )
        # The ValueError must come from the ImportError branch.
        assert isinstance(excinfo.value.__context__, ImportError)
        # The failed call must not leave a cache entry behind.
        assert len(client_cache_module._client_cache) == 0

        # With httpx importable again, the identical call constructs
        # and caches a client normally.
        client = call_getter()
        assert isinstance(client, openai_cls)
        assert len(client_cache_module._client_cache) == 1
        assert call_getter() is client

    @pytest.mark.parametrize(
        "getter,httpx_attr,api_key",
        [
            (get_openai_client, "Client", "test-key-configs-sync"),
            (
                get_async_openai_client,
                "AsyncClient",
                "test-key-configs-async",
            ),
        ],
        ids=["sync", "async"],
    )
    def test_distinct_http_client_configs_get_distinct_clients(
        self, monkeypatch, getter, httpx_attr, api_key
    ):
        """Same API key but different http_client_config: no client reuse.

        http_client_config is part of the cache key, so each distinct
        config must get its own client and its own httpx transport.
        """
        created_http_clients: list[Any] = []
        real_http_client_cls = getattr(httpx, httpx_attr)

        class TrackingHttpClient(real_http_client_cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created_http_clients.append(self)

        monkeypatch.setattr(httpx, httpx_attr, TrackingHttpClient)

        client1 = getter(
            api_key=api_key,
            http_client_config={"timeout": 1.0},
        )
        client2 = getter(
            api_key=api_key,
            http_client_config={"timeout": 2.0},
        )

        assert client1 is not client2
        assert len(created_http_clients) == 2

        # Each distinct config keeps its own cache entry: repeat calls
        # are hits and construct no further transports.
        assert getter(api_key=api_key, http_client_config={"timeout": 1.0}) is client1
        assert getter(api_key=api_key, http_client_config={"timeout": 2.0}) is client2
        assert len(created_http_clients) == 2

    def test_prune_cache_negative_max_age_raises(self):
        """Test that negative max_age_seconds raises ValueError."""
        with pytest.raises(ValueError, match="max_age_seconds must be non-negative"):
            prune_cache(-1.0)

    def test_prune_cache_zero_max_age(self, monkeypatch):
        """Test that max_age_seconds=0 evicts all entries."""
        fake_now = [4000.0]
        monkeypatch.setattr(client_cache_module.time, "monotonic", lambda: fake_now[0])

        get_openai_client(api_key="test-key-zero-a")
        get_openai_client(api_key="test-key-zero-b")

        # Advance time by the smallest amount so entries are older than 0.
        fake_now[0] += 0.001

        removed = prune_cache(0.0)
        assert removed == 2

    def test_mixed_client_types(self):
        """Test that different client types are cached separately."""
        api_key = "same-key-for-all"

        openai_client = get_openai_client(api_key=api_key)
        groq_client = get_groq_client(api_key=api_key)
        cerebras_client = get_cerebras_client(api_key=api_key)

        # All should be different objects despite same API key
        assert openai_client is not groq_client
        assert openai_client is not cerebras_client
        assert groq_client is not cerebras_client

    # Integration tests with OpenAIGPT

    def test_openai_gpt_client_reuse(self):
        """Test that multiple OpenAIGPT instances reuse clients."""
        config = OpenAIGPTConfig(
            api_key="test-key-123",
            chat_model="gpt-4",
        )

        # Create two instances with same config
        gpt1 = OpenAIGPT(config)
        gpt2 = OpenAIGPT(config)

        # They should share the same client instances
        assert gpt1.client is gpt2.client
        assert gpt1.async_client is gpt2.async_client

    def test_openai_gpt_different_config(self):
        """Test that different configs create different clients."""
        config1 = OpenAIGPTConfig(
            api_key="test-key-1",
            chat_model="gpt-4",
        )
        config2 = OpenAIGPTConfig(
            api_key="test-key-2",
            chat_model="gpt-4",
        )

        gpt1 = OpenAIGPT(config1)
        gpt2 = OpenAIGPT(config2)

        # Different API keys should result in different clients
        assert gpt1.client is not gpt2.client
        assert gpt1.async_client is not gpt2.async_client

    def test_use_cached_client_flag(self):
        """Test that use_cached_client config works correctly."""
        # With caching enabled (default)
        config_cached = OpenAIGPTConfig(
            api_key="test-key",
            chat_model="gpt-4",
            use_cached_client=True,
        )

        gpt1 = OpenAIGPT(config_cached)
        gpt2 = OpenAIGPT(config_cached)
        assert gpt1.client is gpt2.client

        # With caching disabled
        config_no_cache = OpenAIGPTConfig(
            api_key="test-key",
            chat_model="gpt-4",
            use_cached_client=False,
        )

        gpt3 = OpenAIGPT(config_no_cache)
        gpt4 = OpenAIGPT(config_no_cache)

        # Each instance should have its own client
        assert gpt3.client is not gpt4.client
        assert gpt3.client is not gpt1.client

    @pytest.mark.parametrize("use_cached_client", [True, False])
    def test_concurrent_client_sharing(self, use_cached_client):
        """Test that multiple OpenAIGPT instances share clients correctly."""
        # Create 10 OpenAIGPT instances with same config
        config = OpenAIGPTConfig(
            api_key="test-key-concurrent",
            chat_model="gpt-4",
            use_cached_client=use_cached_client,
        )

        instances = [OpenAIGPT(config) for _ in range(10)]

        if use_cached_client:
            # With caching, they should all share the same sync and async clients
            for i in range(1, 10):
                assert instances[0].client is instances[i].client
                assert instances[0].async_client is instances[i].async_client
        else:
            # Without caching, each should have its own clients
            for i in range(1, 10):
                assert instances[0].client is not instances[i].client
                assert instances[0].async_client is not instances[i].async_client

        # Verify the client is an OpenAI client instance
        assert instances[0].client.__class__.__name__ == "OpenAI"
        assert instances[0].async_client.__class__.__name__ == "AsyncOpenAI"

        # Create instance with different API key - should always get different client
        config_diff = OpenAIGPTConfig(
            api_key="different-test-key",
            chat_model="gpt-4",
            use_cached_client=use_cached_client,
        )
        instance_diff = OpenAIGPT(config_diff)

        # Different API keys should always result in different clients
        assert instance_diff.client is not instances[0].client
        assert instance_diff.async_client is not instances[0].async_client

    @pytest.mark.asyncio
    @pytest.mark.parametrize("use_cached_client", [True, False])
    async def test_concurrent_async_achat(self, use_cached_client):
        """Test that multiple OpenAIGPT instances can make concurrent achat calls."""
        import asyncio

        # Create 10 OpenAIGPT instances with same config
        # API key will be picked up from environment
        config = OpenAIGPTConfig(
            chat_model="gpt-4o-mini",  # Use a cheaper model for testing
            use_cached_client=use_cached_client,
            max_output_tokens=10,  # Keep responses short for testing
        )

        instances = [OpenAIGPT(config) for _ in range(10)]

        # Verify client sharing based on use_cached_client flag
        if use_cached_client:
            # With caching, they should all share the same async client
            for i in range(1, 10):
                assert instances[0].async_client is instances[i].async_client
        else:
            # Without caching, each should have its own client
            for i in range(1, 10):
                assert instances[0].async_client is not instances[i].async_client

        # Define async function to make an achat request
        async def make_achat_request(gpt_instance, idx):
            """Make an async achat request."""
            try:
                response = await gpt_instance.achat(
                    messages=f"what comes after {idx}?",
                    max_tokens=10,
                )
                return idx, "success", response.message
            except Exception as e:
                return idx, "error", f"{type(e).__name__}: {str(e)}"

        # Run all requests concurrently
        tasks = [make_achat_request(inst, i) for i, inst in enumerate(instances)]
        results = await asyncio.gather(*tasks)

        # Verify all requests completed
        assert len(results) == 10

        # Verify they all succeeded (works with or without caching)
        for idx, (req_idx, status, response) in enumerate(results):
            assert req_idx == idx
            assert status == "success"
            # Response should contain the number
            assert str(idx + 1) in response or "zero" in response.lower()

    def test_model_prefix_client_selection(self):
        """Test that different model prefixes activate the correct client types."""
        import os

        # Get the current OPENAI_API_KEY env var value to restore later
        original_openai_key = os.environ.get("OPENAI_API_KEY")

        # Set to dummy value to trigger provider-specific client logic
        if original_openai_key:
            del os.environ["OPENAI_API_KEY"]

        try:
            # Test Groq client
            from langroid.utils.configuration import settings

            original_chat_model = settings.chat_model
            settings.chat_model = ""  # Clear any global override

            groq_config = OpenAIGPTConfig(
                api_key="xxx",  # Use DUMMY_API_KEY value
                chat_model="groq/llama3-8b-8192",
                use_cached_client=True,
            )
            groq_gpt = OpenAIGPT(groq_config)
            assert groq_gpt.client.__class__.__name__ == "Groq"
            assert groq_gpt.async_client.__class__.__name__ == "AsyncGroq"
            assert groq_gpt.is_groq is True
            # Model name should have prefix stripped
            assert groq_gpt.config.chat_model == "llama3-8b-8192"

            # Test standard OpenAI models
            openai_config = OpenAIGPTConfig(
                api_key="test-key",
                chat_model="gpt-4",
                use_cached_client=True,
            )
            openai_gpt = OpenAIGPT(openai_config)
            assert openai_gpt.client.__class__.__name__ == "OpenAI"
            assert openai_gpt.config.chat_model == "gpt-4"

        finally:
            # Restore original settings
            settings.chat_model = original_chat_model
            # Restore original OPENAI_API_KEY
            if original_openai_key:
                os.environ["OPENAI_API_KEY"] = original_openai_key
