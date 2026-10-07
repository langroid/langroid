"""
Tests for the pro-active rate limiter (`langroid.language_models.rate_limiter`)
and its wiring into the OpenAI chat-completion call sites.

No real LLM calls: the integration tests run against a local stub server that
emulates OpenAI's leaky-bucket rate limiting, reports the standard
`x-ratelimit-*` headers, and returns 429 when its bucket is empty. So these
tests cost nothing and are deterministic about the *bound* they check (no 429s
were served), while being tolerant about absolute timings.
"""

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Tuple

import pytest

from langroid.language_models.base import RetryParams
from langroid.language_models.openai_gpt import OpenAIGPT, OpenAIGPTConfig
from langroid.language_models.rate_limiter import (
    RateLimitConfig,
    RateLimiter,
    RateLimitSnapshot,
    get_rate_limiter,
    parse_reset_duration,
    rate_limit_error_headers,
    reset_rate_limiters,
)
from langroid.utils.configuration import Settings, temporary_settings


@pytest.fixture(autouse=True)
def _clean_limiter_registry():
    reset_rate_limiters()
    yield
    reset_rate_limiters()


# --------------------------------------------------------------------------- #
# header parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("120ms", 0.12),
        ("1s", 1.0),
        ("6m0s", 360.0),
        ("1h2m3s", 3723.0),
        ("7.66s", 7.66),
        ("0s", 0.0),
        ("2", 2.0),  # bare number: seconds
        ("1m", 60.0),
    ],
)
def test_parse_reset_duration(raw, expected):
    assert parse_reset_duration(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", [None, "", "   ", "abc", "10x", "-5s"])
def test_parse_reset_duration_unparseable(raw):
    assert parse_reset_duration(raw) is None


def test_snapshot_derives_refill_rates():
    """`(limit - remaining) / reset` recovers the account's actual limit."""
    # 500 req/min == 8.333 req/s: one request used, so the bucket refills in
    # 1/8.333 == 120ms.
    snap = RateLimitSnapshot.from_headers(
        {
            "X-RateLimit-Limit-Requests": "500",  # mixed case on purpose
            "x-ratelimit-remaining-requests": "499",
            "x-ratelimit-reset-requests": "120ms",
            "x-ratelimit-limit-tokens": "30000",
            "x-ratelimit-remaining-tokens": "29000",
            "x-ratelimit-reset-tokens": "2s",
        }
    )
    assert snap.has_rate_limit_info
    assert snap.request_refill_rate == pytest.approx(500 / 60, rel=1e-2)
    assert snap.token_refill_rate == pytest.approx(500.0)


def test_snapshot_full_bucket_carries_no_rate_info():
    """remaining == limit means reset is 0, which tells us nothing."""
    snap = RateLimitSnapshot.from_headers(
        {
            "x-ratelimit-limit-requests": "500",
            "x-ratelimit-remaining-requests": "500",
            "x-ratelimit-reset-requests": "0s",
        }
    )
    assert snap.has_rate_limit_info
    assert snap.request_refill_rate is None


def test_snapshot_no_rate_limit_headers():
    snap = RateLimitSnapshot.from_headers({"content-type": "application/json"})
    assert not snap.has_rate_limit_info
    assert snap.request_refill_rate is None


# --------------------------------------------------------------------------- #
# limiter pacing, in isolation
# --------------------------------------------------------------------------- #


def _headers_for(limit: int, remaining: int, rate: float) -> Dict[str, str]:
    """Headers a leaky-bucket provider with `rate` req/s would report."""
    return {
        "x-ratelimit-limit-requests": str(limit),
        "x-ratelimit-remaining-requests": str(remaining),
        "x-ratelimit-reset-requests": f"{(limit - remaining) / rate:.6f}s",
    }


@pytest.mark.parametrize("rate", [20.0, 50.0])
def test_interval_tracks_discovered_request_rate(rate):
    """The pacing interval is 1/(discovered rate * (1 - headroom))."""
    limiter = RateLimiter(RateLimitConfig(enabled=True, headroom=0.1))
    assert limiter.stats()["interval"] == pytest.approx(0.05)  # warmup only
    limiter.observe_response(headers=_headers_for(100, 90, rate))
    stats = limiter.stats()
    assert stats["seen_headers"] is True
    assert stats["request_rate"] == pytest.approx(rate, rel=1e-3)
    assert stats["interval"] == pytest.approx(1.0 / (rate * 0.9), rel=1e-3)


def test_observed_send_rate_tracks_discovered_budget():
    """End-to-end on the limiter alone: measured send rate follows the budget.

    Two different discovered budgets must give two proportionally different
    measured send rates -- otherwise the measurement is not actually tracking
    the parameter.
    """
    measured = {}
    n = 15
    for rate in (20.0, 60.0):
        limiter = RateLimiter(RateLimitConfig(enabled=True, headroom=0.1))
        limiter.observe_response(headers=_headers_for(100, 90, rate))
        t0 = time.monotonic()
        for _ in range(n):
            limiter.acquire()
        elapsed = time.monotonic() - t0
        assert limiter.stats()["sends"] == n  # the limiter really ran
        measured[rate] = n / elapsed

    # never faster than the discovered budget
    for rate, obs in measured.items():
        assert obs <= rate, f"sent {obs:.1f}/s against a {rate}/s budget"
    # and not absurdly slower than the budget minus headroom
    for rate, obs in measured.items():
        assert obs >= 0.5 * rate * 0.9
    # the metric tracks the parameter: 3x the budget, ~3x the send rate
    assert measured[60.0] > 2.0 * measured[20.0]


def test_disabled_limiter_does_not_wait():
    """A limiter that never observes anything and has no warmup never sleeps."""
    limiter = RateLimiter(RateLimitConfig(enabled=True, warmup_interval=0.0))
    t0 = time.monotonic()
    for _ in range(50):
        limiter.acquire()
    elapsed = time.monotonic() - t0
    assert elapsed < 0.1
    stats = limiter.stats()
    assert stats["sends"] == 50
    assert stats["waits"] == 0


def test_low_remaining_budget_stalls():
    """Below the reserve, the limiter stalls until the bucket has refilled."""
    limiter = RateLimiter(
        RateLimitConfig(enabled=True, min_remaining_requests=1, warmup_interval=0.0)
    )
    # 10 req/s budget, nothing left: need ~2 refills (0.2s) to clear reserve.
    limiter.observe_response(headers=_headers_for(100, 0, 10.0))
    t0 = time.monotonic()
    limiter.acquire()
    waited = time.monotonic() - t0
    assert 0.1 <= waited <= 1.0


def test_max_wait_caps_the_sleep():
    """A tiny budget must not make a caller hang for its full turn."""
    limiter = RateLimiter(
        RateLimitConfig(enabled=True, max_wait=0.05, warmup_interval=0.0)
    )
    # 0.01 req/s -> ~111s between sends; without the cap this test would hang.
    limiter.observe_response(headers=_headers_for(100, 90, 0.01))
    t0 = time.monotonic()
    for _ in range(3):
        limiter.acquire()
    waited = time.monotonic() - t0
    assert waited < 1.0
    stats = limiter.stats()
    # the first send is free; the next two each hit the cap
    assert stats["capped_waits"] == 2
    assert stats["total_wait"] == pytest.approx(0.10, abs=1e-6)
    # the cap did not corrupt the discovered pacing
    assert stats["interval"] == pytest.approx(1.0 / (0.01 * 0.9), rel=1e-3)


def test_token_budget_paces_once_usage_is_known():
    limiter = RateLimiter(RateLimitConfig(enabled=True, headroom=0.0))
    headers = {
        "x-ratelimit-limit-tokens": "1000",
        "x-ratelimit-remaining-tokens": "900",
        "x-ratelimit-reset-tokens": "1s",  # 100 tokens/s
    }
    limiter.observe_response(headers=headers)
    # no per-request token estimate yet -> no token pacing
    assert limiter.stats()["avg_tokens"] is None
    limiter.observe_response(tokens_used=50)
    stats = limiter.stats()
    assert stats["token_rate"] == pytest.approx(100.0)
    assert stats["avg_tokens"] == pytest.approx(50.0)
    # 50 tokens per request at 100 tokens/s -> 0.5s between sends
    assert stats["interval"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# header-free fallback (AIMD)
# --------------------------------------------------------------------------- #


def test_headerless_fallback_backs_off_and_recovers():
    cfg = RateLimitConfig(
        enabled=True,
        warmup_interval=0.0,
        error_interval=0.05,
        backoff_factor=2.0,
        recovery_factor=0.5,
    )
    limiter = RateLimiter(cfg)
    assert limiter.stats()["interval"] == 0.0

    limiter.observe_rate_limit_error()  # no headers at all
    first = limiter.stats()["fallback_interval"]
    assert first == pytest.approx(0.05)

    limiter.observe_rate_limit_error()
    second = limiter.stats()["fallback_interval"]
    assert second == pytest.approx(0.10)
    assert limiter.stats()["rate_limit_errors"] == 2

    # recovers on success
    limiter.observe_response()
    assert limiter.stats()["fallback_interval"] == pytest.approx(0.05)
    for _ in range(10):
        limiter.observe_response()
    assert limiter.stats()["fallback_interval"] == 0.0


def test_headerless_fallback_respects_max_interval():
    limiter = RateLimiter(
        RateLimitConfig(enabled=True, error_interval=0.5, max_interval=1.0)
    )
    for _ in range(10):
        limiter.observe_rate_limit_error()
    assert limiter.stats()["fallback_interval"] == pytest.approx(1.0)


def test_rate_limit_error_headers_classification():
    class NotRateLimit(Exception):
        status_code = 500

    class RateLimitError(Exception):
        pass

    class WithHeaders(Exception):
        status_code = 429

        class _Resp:
            status_code = 429
            headers = {"retry-after": "3s"}

        response = _Resp()

    assert rate_limit_error_headers(NotRateLimit()) is None
    assert rate_limit_error_headers(ValueError("boom")) is None
    # a 429 with no headers is an empty mapping, which is NOT None
    assert rate_limit_error_headers(RateLimitError()) == {}
    assert rate_limit_error_headers(WithHeaders()) == {"retry-after": "3s"}


def test_retry_after_header_stalls_the_limiter():
    limiter = RateLimiter(RateLimitConfig(enabled=True, warmup_interval=0.0))
    limiter.observe_rate_limit_error({"retry-after": "150ms"})
    t0 = time.monotonic()
    limiter.acquire()
    assert time.monotonic() - t0 >= 0.1


# --------------------------------------------------------------------------- #
# sharing across cloned agents
# --------------------------------------------------------------------------- #


def test_limiters_are_shared_by_key():
    cfg = RateLimitConfig(enabled=True)
    a = get_rate_limiter("openai::gpt-4o-mini", cfg)
    b = get_rate_limiter("openai::gpt-4o-mini", cfg)
    c = get_rate_limiter("openai::gpt-4.1", cfg)
    assert a is b
    assert a is not c
    a.acquire()
    assert b.stats()["sends"] == 1
    assert c.stats()["sends"] == 0


def test_cloned_llms_share_one_limiter():
    """Two OpenAIGPT instances on the same (api_base, model) pace together."""
    cfg = OpenAIGPTConfig(
        chat_model="gpt-4o-mini",
        api_key="test",
        rate_limit=RateLimitConfig(enabled=True),
    )
    llm1 = OpenAIGPT(cfg)
    llm2 = OpenAIGPT(cfg.model_copy(deep=True))
    assert llm1._rate_limiter() is llm2._rate_limiter()


def test_limiter_is_off_by_default():
    cfg = OpenAIGPTConfig(chat_model="gpt-4o-mini", api_key="test")
    assert cfg.rate_limit.enabled is False
    assert OpenAIGPT(cfg)._rate_limiter() is None


# --------------------------------------------------------------------------- #
# integration against a stub server that really enforces a rate limit
# --------------------------------------------------------------------------- #


class _LeakyBucket:
    """Emulates OpenAI's request rate limit and the headers it reports."""

    def __init__(self, limit: int, refill_per_sec: float) -> None:
        self.limit = limit
        self.refill = refill_per_sec
        self._tokens = float(limit)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def take(self) -> Tuple[bool, Dict[str, str]]:
        with self._lock:
            now = time.monotonic()
            self._tokens = min(
                float(self.limit), self._tokens + (now - self._last) * self.refill
            )
            self._last = now
            allowed = self._tokens >= 1.0
            if allowed:
                self._tokens -= 1.0
            remaining = int(self._tokens)
            # `reset` is the time until the bucket is full again, computed from
            # the same integer `remaining` we report, so that
            # (limit - remaining) / reset == refill exactly.
            reset = (self.limit - remaining) / self.refill
            return allowed, {
                "x-ratelimit-limit-requests": str(self.limit),
                "x-ratelimit-remaining-requests": str(remaining),
                "x-ratelimit-reset-requests": f"{reset:.6f}s",
            }


_COMPLETION = {
    "id": "chatcmpl-stub",
    "object": "chat.completion",
    "created": 1,
    "model": "stub",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
}

_CHUNKS = [
    {
        "id": "chatcmpl-stub",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "stub",
        "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": None}],
    },
    {
        "id": "chatcmpl-stub",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "stub",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        # langroid asks for stream_options={"include_usage": True}, so a real
        # provider reports usage in a trailing chunk like this one.
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    },
]

# token budget the stub reports alongside the request budget, when asked
_TOKEN_HEADERS = {
    "x-ratelimit-limit-tokens": "1000",
    "x-ratelimit-remaining-tokens": "900",
    "x-ratelimit-reset-tokens": "1s",  # 100 tokens/s
}


class _StubServer:
    """Local OpenAI-compatible endpoint with a real, enforced rate limit."""

    def __init__(
        self, limit: int, refill_per_sec: float, token_headers: bool = False
    ) -> None:
        self.bucket = _LeakyBucket(limit, refill_per_sec)
        self.token_headers = token_headers
        self.ok_count = 0
        self.rejected_count = 0
        self._count_lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                pass

            def _send(
                self, status: int, payload: bytes, headers: Dict[str, str], ctype: str
            ) -> None:
                self.send_response(status)
                self.send_header("content-type", ctype)
                self.send_header("content-length", str(len(payload)))
                for k, v in headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                allowed, rl_headers = server.bucket.take()
                if server.token_headers:
                    rl_headers = {**rl_headers, **_TOKEN_HEADERS}
                if not allowed:
                    with server._count_lock:
                        server.rejected_count += 1
                    payload = json.dumps(
                        {"error": {"message": "rate limit", "type": "rate_limit"}}
                    ).encode()
                    self._send(429, payload, rl_headers, "application/json")
                    return
                with server._count_lock:
                    server.ok_count += 1
                if body.get("stream"):
                    chunks = b"".join(
                        b"data: " + json.dumps(c).encode() + b"\n\n" for c in _CHUNKS
                    )
                    chunks += b"data: [DONE]\n\n"
                    self._send(200, chunks, rl_headers, "text/event-stream")
                    return
                self._send(
                    200,
                    json.dumps(_COMPLETION).encode(),
                    rl_headers,
                    "application/json",
                )

        # Threading server with daemon threads: a single-threaded HTTPServer
        # blocks in shutdown() while a keep-alive connection is still open.
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def api_base(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


def _llm(server: _StubServer, enabled: bool, key: str, **rl: Any) -> OpenAIGPT:
    return OpenAIGPT(
        OpenAIGPTConfig(
            chat_model="gpt-4o-mini",
            api_base=server.api_base,
            api_key="test",
            stream=False,
            timeout=30,
            retry_params=RetryParams(max_retries=2, initial_delay=0.01),
            rate_limit=RateLimitConfig(enabled=enabled, share_key=key, **rl),
        )
    )


def _sequential_calls(llm: OpenAIGPT, n: int) -> int:
    """Make `n` chat calls, swallowing failures; return how many succeeded."""
    ok = 0
    for i in range(n):
        try:
            llm.chat(f"say ok {i}", max_tokens=5)
            ok += 1
        except Exception:
            pass
    return ok


@pytest.mark.parametrize("refill", [20.0, 60.0])
def test_limiter_respects_a_real_enforced_rate_limit(refill):
    """With the limiter ON the stub never has to reject a request.

    And the achieved send rate tracks the stub's (discovered) budget: see
    `test_limiter_disabled_trips_the_rate_limit` for the counter-verification
    that the same workload DOES get rejected with the limiter off.
    """
    server = _StubServer(limit=8, refill_per_sec=refill)
    n = 24
    try:
        with temporary_settings(Settings(cache=False, cache_type="none")):
            llm = _llm(server, enabled=True, key=f"stub-on-{refill}")
            t0 = time.monotonic()
            ok = _sequential_calls(llm, n)
            elapsed = time.monotonic() - t0
    finally:
        server.close()

    limiter = get_rate_limiter(f"stub-on-{refill}")
    stats = limiter.stats()
    # (a) did it run?
    assert stats["sends"] == n, stats
    assert stats["seen_headers"] is True, stats
    assert stats["request_rate"] == pytest.approx(refill, rel=0.2), stats
    # (b) what did it find?
    assert ok == n, f"{n - ok} calls failed"
    assert server.ok_count == n
    assert server.rejected_count == 0, "the limiter let the budget be exceeded"
    send_rate = n / elapsed
    assert send_rate <= refill, f"sent {send_rate:.1f}/s against {refill}/s"


def test_limiter_disabled_trips_the_rate_limit():
    """Counter-verification: the same workload is rejected with the limiter off.

    This is what makes the test above non-vacuous -- it shows the stub really
    does enforce a limit that an unpaced sender blows through.
    """
    server = _StubServer(limit=8, refill_per_sec=20.0)
    try:
        with temporary_settings(Settings(cache=False, cache_type="none")):
            llm = _llm(server, enabled=False, key="stub-off")
            _sequential_calls(llm, 24)
    finally:
        server.close()

    assert (
        server.rejected_count > 0
    ), "stub served no 429 even unpaced; the limiter test would be vacuous"


def test_limiter_keeps_headers_working_for_streaming():
    """The raw-response path must not break streaming."""
    server = _StubServer(limit=50, refill_per_sec=100.0)
    try:
        with temporary_settings(Settings(cache=False, cache_type="none")):
            llm = OpenAIGPT(
                OpenAIGPTConfig(
                    chat_model="gpt-4o-mini",
                    api_base=server.api_base,
                    api_key="test",
                    stream=True,
                    timeout=30,
                    rate_limit=RateLimitConfig(enabled=True, share_key="stub-stream"),
                )
            )
            response = llm.chat("say ok", max_tokens=5)
    finally:
        server.close()

    assert "ok" in response.message
    stats = get_rate_limiter("stub-stream").stats()
    assert stats["sends"] == 1
    assert stats["seen_headers"] is True, "headers were lost on the streaming path"


def test_async_concurrent_calls_respect_the_rate_limit():
    """The async path paces a concurrent batch, including its first wave."""
    server = _StubServer(limit=8, refill_per_sec=60.0)
    n = 24

    async def run() -> List[Any]:
        llm = _llm(server, enabled=True, key="stub-async", warmup_interval=0.05)
        return await asyncio.gather(
            *[llm.achat(f"say ok {i}", max_tokens=5) for i in range(n)],
            return_exceptions=True,
        )

    try:
        with temporary_settings(Settings(cache=False, cache_type="none")):
            results = asyncio.run(run())
    finally:
        server.close()

    errors = [r for r in results if isinstance(r, BaseException)]
    stats = get_rate_limiter("stub-async").stats()
    assert stats["sends"] == n, stats
    assert stats["seen_headers"] is True, stats
    assert not errors, f"{len(errors)} async calls failed: {errors[:2]}"
    assert server.rejected_count == 0, "concurrent batch exceeded the budget"


def test_streaming_responses_feed_the_token_estimate():
    """Usage arrives in a trailing chunk; the limiter must still learn it.

    Without this, a streaming workload -- which is langroid's default -- would
    never engage token-budget pacing, since the stream object itself carries no
    `usage`.
    """
    server = _StubServer(limit=50, refill_per_sec=100.0, token_headers=True)
    try:
        with temporary_settings(Settings(cache=False, cache_type="none")):
            llm = OpenAIGPT(
                OpenAIGPTConfig(
                    chat_model="gpt-4o-mini",
                    api_base=server.api_base,
                    api_key="test",
                    stream=True,
                    timeout=30,
                    rate_limit=RateLimitConfig(
                        enabled=True, share_key="stub-stream-tokens", headroom=0.0
                    ),
                )
            )
            response = llm.chat("say ok", max_tokens=5)
    finally:
        server.close()

    assert "ok" in response.message
    stats = get_rate_limiter("stub-stream-tokens").stats()
    assert stats["token_rate"] == pytest.approx(100.0), stats
    assert stats["avg_tokens"] == pytest.approx(12.0), stats
    # 12 tokens/request against 100 tokens/s -> 0.12s between sends
    assert stats["interval"] == pytest.approx(0.12, rel=1e-3), stats


# --------------------------------------------------------------------------- #
# cooldowns must reach callers that are already queued
# --------------------------------------------------------------------------- #


def test_cooldown_holds_back_already_queued_callers():
    """A 429 discovered mid-flight must stall callers already asleep.

    Reservations are handed out before the limiter knows what the response will
    say. If a queued caller only honours its original reservation, a whole
    concurrent wave sails straight through the cooldown.
    """
    limiter = RateLimiter(
        RateLimitConfig(enabled=True, warmup_interval=0.02, max_wait=5.0)
    )
    n = 6
    sent_at: List[float] = []
    lock = threading.Lock()
    t0 = time.monotonic()

    def worker() -> None:
        limiter.acquire()
        with lock:
            sent_at.append(time.monotonic() - t0)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    time.sleep(0.01)
    limiter.observe_rate_limit_error({"retry-after": "0.4s"})
    for t in threads:
        t.join(timeout=10)
    assert all(not t.is_alive() for t in threads)

    assert len(sent_at) == n
    # the first caller may already have gone before the 429 was observed;
    # everyone still queued at that moment must wait out the cooldown
    held = [t for t in sent_at if t >= 0.01]
    assert len(held) >= n - 1, sent_at
    assert min(held) >= 0.35, f"cooldown bypassed by a queued caller: {sent_at}"
    assert limiter.stats()["rate_limit_errors"] == 1


def test_cooldown_wait_is_bounded_by_max_wait():
    """The gate must not turn into an unbounded hang."""
    limiter = RateLimiter(
        RateLimitConfig(enabled=True, warmup_interval=0.0, max_wait=0.1)
    )
    limiter.observe_rate_limit_error({"retry-after": "30s"})
    t0 = time.monotonic()
    limiter.acquire()
    waited = time.monotonic() - t0
    # max_wait is 0.1s: a 30s retry-after must not become a 30s hold
    assert waited <= 0.5, waited
    assert limiter.stats()["rate_limit_errors"] == 1
