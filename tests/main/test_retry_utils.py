"""
Tests for the retry decorators in langroid.language_models.utils.

Focus: non-retryable 4xx client errors (e.g. a 404 from a retired model on the
direct OpenAI-compatible API, such as Gemini) must fail fast rather than being
retried with exponential backoff. See the type-based guard in
`retry_with_exponential_backoff` / `async_retry_with_exponential_backoff`.
"""

import httpx
import openai
import pytest

from langroid.language_models.utils import (
    async_retry_with_exponential_backoff,
    retry_with_exponential_backoff,
)


def _make_openai_error(exc_cls: type, status: int) -> openai.APIStatusError:
    """Build a *native* OpenAI-SDK error, whose str() is like
    'Error code: 404 - ...' and does NOT contain the class name."""
    request = httpx.Request(
        "POST",
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    )
    response = httpx.Response(
        status,
        request=request,
        json={"error": {"code": status, "message": "boom", "status": "NOT_FOUND"}},
    )
    return exc_cls(f"Error code: {status}", response=response, body=None)


# 4xx client errors that must NOT be retried
NON_RETRYABLE = [
    (openai.BadRequestError, 400),
    (openai.AuthenticationError, 401),
    (openai.PermissionDeniedError, 403),
    (openai.NotFoundError, 404),
    (openai.UnprocessableEntityError, 422),
]


@pytest.mark.parametrize("exc_cls,status", NON_RETRYABLE)
def test_sync_non_retryable_fails_fast(exc_cls, status):
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise _make_openai_error(exc_cls, status)

    wrapped = retry_with_exponential_backoff(
        fn, max_retries=3, initial_delay=0.0, jitter=False
    )
    with pytest.raises(exc_cls):
        wrapped()

    # called exactly once => no retries
    assert calls["n"] == 1


@pytest.mark.parametrize("exc_cls,status", NON_RETRYABLE)
@pytest.mark.asyncio
async def test_async_non_retryable_fails_fast(exc_cls, status):
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        raise _make_openai_error(exc_cls, status)

    wrapped = async_retry_with_exponential_backoff(
        fn, max_retries=3, initial_delay=0.0, jitter=False
    )
    with pytest.raises(exc_cls):
        await wrapped()

    assert calls["n"] == 1


def test_sync_rate_limit_is_retried():
    """Regression guard: retryable errors (429) still retry until max_retries."""
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise _make_openai_error(openai.RateLimitError, 429)

    wrapped = retry_with_exponential_backoff(
        fn, max_retries=2, initial_delay=0.0, jitter=False
    )
    with pytest.raises(Exception, match="Maximum number of retries"):
        wrapped()

    # initial attempt + max_retries retries
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_async_rate_limit_is_retried():
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        raise _make_openai_error(openai.RateLimitError, 429)

    wrapped = async_retry_with_exponential_backoff(
        fn, max_retries=2, initial_delay=0.0, jitter=False
    )
    with pytest.raises(Exception, match="Maximum number of retries"):
        await wrapped()

    assert calls["n"] == 3
