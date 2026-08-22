"""
Tests for rotating/short-lived credential support via
`OpenAIGPTConfig.api_key_provider` (GitHub issue #1080).

A callable API-key provider is resolved per-request by the OpenAI client
(so tokens never go stale), and is excluded from the client-cache key
(so rotating tokens never grow the client cache).
"""

import itertools
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
import pytest

import langroid.language_models.client_cache as client_cache_module
from langroid.language_models.client_cache import (
    _clear_cache,
    get_async_openai_client,
    get_openai_client,
)
from langroid.language_models.openai_gpt import OpenAIGPT, OpenAIGPTConfig
from langroid.utils.configuration import settings

CHAT_COMPLETION_JSON: Dict[str, Any] = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-4",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "mock-response"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def _make_mock_clients(
    seen_auth_headers: List[Optional[str]],
) -> Tuple[httpx.Client, httpx.AsyncClient]:
    """Return (sync, async) httpx clients that record Authorization headers."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth_headers.append(request.headers.get("authorization"))
        return httpx.Response(200, json=CHAT_COMPLETION_JSON)

    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport), httpx.AsyncClient(transport=transport)


def _rotating_provider(prefix: str = "tok") -> Callable[[], str]:
    """A provider that returns a fresh token on every call."""
    counter = itertools.count(1)

    def provider() -> str:
        return f"{prefix}-{next(counter)}"

    return provider


def _make_llm(
    provider: Optional[Callable[[], str]],
    seen_auth_headers: List[Optional[str]],
    use_cached_client: bool = False,
    **config_kwargs: Any,
) -> OpenAIGPT:
    """OpenAIGPT wired to a mock transport that records auth headers."""
    config = OpenAIGPTConfig(
        chat_model="gpt-4",
        api_key_provider=provider,
        http_client_factory=lambda: _make_mock_clients(seen_auth_headers),
        use_cached_client=use_cached_client,
        stream=False,
        cache_config=None,  # disable response caching (no Redis in tests)
        **config_kwargs,
    )
    return OpenAIGPT(config)


class TestRotatingApiKeys:
    def setup_method(self) -> None:
        _clear_cache()

    @pytest.mark.parametrize("use_cached_client", [True, False])
    def test_sync_chat_rotates_tokens(self, use_cached_client: bool) -> None:
        """Each sync request carries a freshly-resolved bearer token."""
        seen: List[Optional[str]] = []
        llm = _make_llm(
            _rotating_provider(),
            seen,
            use_cached_client=use_cached_client,
        )

        llm.chat("hello", max_tokens=5)
        llm.chat("hello again", max_tokens=5)

        assert seen == ["Bearer tok-1", "Bearer tok-2"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("use_cached_client", [True, False])
    async def test_async_chat_rotates_tokens(self, use_cached_client: bool) -> None:
        """Each async request carries a freshly-resolved bearer token."""
        seen: List[Optional[str]] = []
        llm = _make_llm(
            _rotating_provider(),
            seen,
            use_cached_client=use_cached_client,
        )

        await llm.achat("hello", max_tokens=5)
        await llm.achat("hello again", max_tokens=5)

        assert seen == ["Bearer tok-1", "Bearer tok-2"]

    def test_provider_wins_over_static_api_key(self) -> None:
        """When both api_key and api_key_provider are set, the provider wins."""
        seen: List[Optional[str]] = []
        llm = _make_llm(
            _rotating_provider("dynamic"),
            seen,
            api_key="static-key",
        )

        llm.chat("hello", max_tokens=5)

        assert seen == ["Bearer dynamic-1"]

    def test_static_api_key_still_used_without_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: without a provider, the static api_key is used as-is."""
        # the config reads OPENAI_API_KEY via pydantic-settings, so drop it
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        seen: List[Optional[str]] = []
        llm = _make_llm(None, seen, api_key="static-key")

        llm.chat("hello", max_tokens=5)

        assert seen == ["Bearer static-key"]

    def test_provider_identity_survives_config_copy(self) -> None:
        """The provider must not be deep-copied by OpenAIGPT's config copy:
        it may hold non-copyable state, and the client cache is keyed on its
        identity."""

        class TokenProvider:
            def __init__(self) -> None:
                self._lock = threading.Lock()  # not deep-copyable

            def __call__(self) -> str:
                with self._lock:
                    return "tok-instance"

        provider = TokenProvider()
        config = OpenAIGPTConfig(
            chat_model="gpt-4",
            api_key_provider=provider,
            use_cached_client=True,
            cache_config=None,
        )

        llm1 = OpenAIGPT(config)
        llm2 = OpenAIGPT(config)

        assert llm1.config.api_key_provider is provider
        assert llm1.client is llm2.client
        assert llm1.async_client is llm2.async_client
        # exactly one sync + one async entry
        assert len(client_cache_module._client_cache) == 2

    @pytest.mark.asyncio
    async def test_async_provider_runs_off_event_loop(self) -> None:
        """The sync provider is run in a worker thread, so a blocking token
        refresh cannot stall the event loop."""
        calling_threads: List[threading.Thread] = []
        counter = itertools.count(1)

        def provider() -> str:
            calling_threads.append(threading.current_thread())
            return f"tok-{next(counter)}"

        seen: List[Optional[str]] = []
        llm = _make_llm(provider, seen)

        await llm.achat("hello", max_tokens=5)

        assert seen == ["Bearer tok-1"]
        assert len(calling_threads) == 1
        assert calling_threads[0] is not threading.current_thread()

    def test_rotation_does_not_grow_client_cache(self) -> None:
        """
        Rotating tokens must not create new cache entries: the cache is keyed
        on the provider's identity, not on any token value (issue #1080's
        unbounded-growth failure mode).
        """
        provider = _rotating_provider()
        config = OpenAIGPTConfig(
            chat_model="gpt-4",
            api_key_provider=provider,
            use_cached_client=True,
            cache_config=None,
        )

        llm1 = OpenAIGPT(config)
        # exercise the provider a few times, as real requests would
        for _ in range(3):
            provider()
        llm2 = OpenAIGPT(config)

        assert llm1.client is llm2.client
        assert llm1.async_client is llm2.async_client
        # exactly one sync + one async entry, despite token rotation
        assert len(client_cache_module._client_cache) == 2

    def test_token_value_not_in_cache_key(self) -> None:
        """Two providers yielding identical tokens still get distinct clients,
        and the same provider always maps to the same cached client."""
        provider_a = lambda: "same-token"  # noqa: E731
        provider_b = lambda: "same-token"  # noqa: E731

        client_a1 = get_openai_client(api_key=provider_a)
        client_a2 = get_openai_client(api_key=provider_a)
        client_b = get_openai_client(api_key=provider_b)

        assert client_a1 is client_a2
        assert client_a1 is not client_b

        aclient_a1 = get_async_openai_client(api_key=provider_a)
        aclient_a2 = get_async_openai_client(api_key=provider_a)
        aclient_b = get_async_openai_client(api_key=provider_b)

        assert aclient_a1 is aclient_a2
        assert aclient_a1 is not aclient_b

    def test_provider_vs_static_key_distinct_cache_entries(self) -> None:
        """A provider-based client never collides with a static-key client."""
        client_static = get_openai_client(api_key="some-key")
        client_provider = get_openai_client(api_key=lambda: "some-key")
        assert client_static is not client_provider

    @pytest.mark.parametrize(
        "chat_model",
        [
            "groq/llama-3.1-8b-instant",
            "cerebras/llama3.1-8b",
            "litellm/ollama/llama3",
        ],
    )
    def test_provider_rejected_for_non_openai_clients(self, chat_model: str) -> None:
        """api_key_provider only applies to the OpenAI-compatible client path."""
        original_chat_model = settings.chat_model
        settings.chat_model = ""  # clear any global model override
        try:
            with pytest.raises(ValueError, match="api_key_provider"):
                OpenAIGPT(
                    OpenAIGPTConfig(
                        chat_model=chat_model,
                        api_key_provider=lambda: "tok",
                        cache_config=None,
                    )
                )
        finally:
            settings.chat_model = original_chat_model

    def test_provider_exceptions_propagate(self) -> None:
        """A failing provider surfaces its error instead of silently sending
        a stale/empty credential."""

        def bad_provider() -> str:
            raise RuntimeError("token fetch failed")

        seen: List[Optional[str]] = []
        llm = _make_llm(bad_provider, seen)

        with pytest.raises(RuntimeError, match="token fetch failed"):
            llm.chat("hello", max_tokens=5)

        assert seen == []  # request never reached the transport

    def test_uncached_clients_use_provider(self) -> None:
        """With use_cached_client=False, each instance gets its own client,
        and rotation still works per request."""
        seen: List[Optional[str]] = []
        provider = _rotating_provider()
        llm1 = _make_llm(provider, seen, use_cached_client=False)
        llm2 = _make_llm(provider, seen, use_cached_client=False)

        assert llm1.client is not llm2.client

        llm1.chat("hi", max_tokens=5)
        llm2.chat("hi", max_tokens=5)
        assert seen == ["Bearer tok-1", "Bearer tok-2"]
