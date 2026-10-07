"""Pro-active rate limiting for OpenAI-compatible chat-completion calls.

This module is standalone: it knows nothing about Langroid agents, and it does
not touch the retry-with-exponential-backoff logic in
`langroid.language_models.utils`, which stays as the reactive fallback.

The limiter discovers the account's actual limits instead of asking the user to
configure them. OpenAI (and most OpenAI-compatible gateways) report the live
budget on every response::

    x-ratelimit-limit-requests / x-ratelimit-remaining-requests
    x-ratelimit-limit-tokens   / x-ratelimit-remaining-tokens
    x-ratelimit-reset-requests / x-ratelimit-reset-tokens

`reset` is the time until the bucket refills to `limit`, so the sustained
refill rate implied by a single response is::

    rate = (limit - remaining) / reset

which is exactly the account's limit for that model, in requests (or tokens)
per second. The limiter paces sends at `rate * (1 - headroom)` so a request is
held briefly rather than rejected.

For providers that send no such headers (local servers, Groq/Cerebras, litellm)
the limiter falls back to an additive-free AIMD scheme: multiply the inter-send
interval up on a 429, decay it down on every success. No prior knowledge of any
limit is needed.

Limiters are shared process-wide by key (see `get_rate_limiter`), so the many
cloned agents of a `run_batch_tasks` job pace against one budget.

Known limitation: until the first response arrives there are no headers to
learn from, so an initial burst of concurrent calls is sent unpaced; pacing
engages from the first observed response onwards.
"""

import asyncio
import logging
import threading
import time
from typing import Any, Dict, Mapping, Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

LIMIT_REQUESTS_HEADER = "x-ratelimit-limit-requests"
REMAINING_REQUESTS_HEADER = "x-ratelimit-remaining-requests"
RESET_REQUESTS_HEADER = "x-ratelimit-reset-requests"
LIMIT_TOKENS_HEADER = "x-ratelimit-limit-tokens"
REMAINING_TOKENS_HEADER = "x-ratelimit-remaining-tokens"
RESET_TOKENS_HEADER = "x-ratelimit-reset-tokens"
RETRY_AFTER_HEADER = "retry-after"

_UNIT_SECONDS = {
    "ms": 1e-3,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
}


def parse_reset_duration(value: Optional[str]) -> Optional[float]:
    """Parse an OpenAI `x-ratelimit-reset-*` duration into seconds.

    The header is a Go-style duration, e.g. `"120ms"`, `"1s"`, `"6m0s"`,
    `"1h2m3s"`, `"7.66s"`. A bare number is read as seconds, which is what
    some OpenAI-compatible gateways send.

    Args:
        value: Raw header value, or None.

    Returns:
        Duration in seconds, or None if `value` is absent or unparseable.
    """
    if value is None:
        return None
    text = value.strip().lower()
    if not text:
        return None
    total = 0.0
    matched = False
    i = 0
    n = len(text)
    while i < n:
        start = i
        while i < n and (text[i].isdigit() or text[i] == "."):
            i += 1
        if i == start:
            return None  # unexpected character where a number was expected
        try:
            amount = float(text[start:i])
        except ValueError:
            return None
        unit_start = i
        while i < n and text[i].isalpha():
            i += 1
        unit = text[unit_start:i]
        if unit == "":
            # bare number: seconds
            total += amount
            matched = True
            continue
        if unit not in _UNIT_SECONDS:
            return None
        total += amount * _UNIT_SECONDS[unit]
        matched = True
    return total if matched else None


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(value.strip()))
    except (ValueError, AttributeError):
        return None


class RateLimitSnapshot(BaseModel):
    """One provider-reported view of the remaining rate-limit budget."""

    limit_requests: Optional[int] = None
    remaining_requests: Optional[int] = None
    reset_requests: Optional[float] = None  # seconds
    limit_tokens: Optional[int] = None
    remaining_tokens: Optional[int] = None
    reset_tokens: Optional[float] = None  # seconds
    retry_after: Optional[float] = None  # seconds

    @classmethod
    def from_headers(cls, headers: Mapping[str, Any]) -> "RateLimitSnapshot":
        """Build a snapshot from HTTP response headers (case-insensitive)."""
        lowered = {str(k).lower(): v for k, v in dict(headers).items()}

        def get(name: str) -> Optional[str]:
            raw = lowered.get(name)
            return None if raw is None else str(raw)

        return cls(
            limit_requests=_parse_int(get(LIMIT_REQUESTS_HEADER)),
            remaining_requests=_parse_int(get(REMAINING_REQUESTS_HEADER)),
            reset_requests=parse_reset_duration(get(RESET_REQUESTS_HEADER)),
            limit_tokens=_parse_int(get(LIMIT_TOKENS_HEADER)),
            remaining_tokens=_parse_int(get(REMAINING_TOKENS_HEADER)),
            reset_tokens=parse_reset_duration(get(RESET_TOKENS_HEADER)),
            retry_after=parse_reset_duration(get(RETRY_AFTER_HEADER)),
        )

    @property
    def has_rate_limit_info(self) -> bool:
        """Did the provider report any rate-limit budget at all?"""
        return any(
            v is not None
            for v in (
                self.limit_requests,
                self.remaining_requests,
                self.limit_tokens,
                self.remaining_tokens,
            )
        )

    @staticmethod
    def _refill_rate(
        limit: Optional[int], remaining: Optional[int], reset: Optional[float]
    ) -> Optional[float]:
        if limit is None or remaining is None or reset is None:
            return None
        used = limit - remaining
        if used <= 0 or reset <= 0:
            # Bucket is full: the response carries no information about the
            # refill rate, so leave whatever we learned earlier in place.
            return None
        return used / reset

    @property
    def request_refill_rate(self) -> Optional[float]:
        """Implied sustained limit, in requests/second (None if unknowable)."""
        return self._refill_rate(
            self.limit_requests, self.remaining_requests, self.reset_requests
        )

    @property
    def token_refill_rate(self) -> Optional[float]:
        """Implied sustained limit, in tokens/second (None if unknowable)."""
        return self._refill_rate(
            self.limit_tokens, self.remaining_tokens, self.reset_tokens
        )


class RateLimitConfig(BaseSettings):
    """Settings for the pro-active rate limiter.

    Every field can be overridden by an env var with the
    `LANGROID_RATE_LIMIT_` prefix, e.g. `LANGROID_RATE_LIMIT_ENABLED=1`.
    """

    # Off by default: with this False the request path is unchanged.
    enabled: bool = False
    # Fraction of the discovered budget to leave unused, as a safety margin.
    headroom: float = 0.1
    # Safety valve: never sleep longer than this for a single request. If the
    # cap binds, the request is sent anyway and reactive retry/backoff handles
    # any 429 that results.
    max_wait: float = 60.0
    # Keep this many requests/tokens of the budget in reserve; when the
    # provider reports less than this remaining, stall until it refills.
    min_remaining_requests: int = 1
    min_remaining_tokens: int = 0
    # Interval used before any rate-limit header has been seen. Without this,
    # the first wave of a concurrent batch would all be sent at once, since
    # there is nothing yet to pace against; set to 0.0 to send it unpaced.
    warmup_interval: float = 0.05
    # Header-free fallback (AIMD on the inter-send interval).
    backoff_factor: float = 2.0  # interval multiplier on a rate-limit error
    recovery_factor: float = 0.9  # interval multiplier on a success
    error_interval: float = 0.05  # interval floor once a 429 has been seen
    max_interval: float = 10.0  # interval ceiling
    # Override the process-wide limiter-sharing key; by default limiters are
    # shared per (api_base, model).
    share_key: Optional[str] = None

    model_config = SettingsConfigDict(env_prefix="LANGROID_RATE_LIMIT_")


class RateLimiter:
    """Paces sends so provider rate limits are approached, not exceeded.

    Thread-safe and asyncio-safe: state is guarded by a `threading.Lock` held
    only for the (non-blocking) bookkeeping, while the wait itself happens
    outside the lock via `time.sleep` or `asyncio.sleep`.

    Slots are handed out as reservations off a shared monotonic clock, so N
    concurrent callers queue rather than all retrying at once.
    """

    def __init__(
        self, config: Optional[RateLimitConfig] = None, name: str = ""
    ) -> None:
        self.config = config or RateLimitConfig()
        self.name = name
        self._lock = threading.Lock()
        self._next_send_at = 0.0  # monotonic time of the next free slot
        self._request_rate: Optional[float] = None  # requests/sec, from headers
        self._token_rate: Optional[float] = None  # tokens/sec, from headers
        self._avg_tokens: Optional[float] = None  # EWMA tokens per request
        self._fallback_interval = 0.0  # header-free AIMD interval
        self._seen_headers = False
        self._sends = 0
        self._waits = 0
        self._total_wait = 0.0
        self._rate_limit_errors = 0
        self._capped_waits = 0

    # ------------------------------------------------------------------ #
    # pacing
    # ------------------------------------------------------------------ #

    def _effective(self, rate: float) -> float:
        return max(rate * (1.0 - self.config.headroom), 1e-9)

    def _interval_locked(self) -> float:
        """Minimum seconds between sends, given what we know so far.

        The header-derived pacing and the header-free AIMD interval are both
        honoured; whichever is more conservative wins. In the common cases only
        one of them is non-zero.
        """
        interval = self._fallback_interval
        if not self._seen_headers:
            interval = max(interval, self.config.warmup_interval)
        if self._request_rate is not None:
            interval = max(interval, 1.0 / self._effective(self._request_rate))
        if self._token_rate is not None and self._avg_tokens:
            interval = max(
                interval, self._avg_tokens / self._effective(self._token_rate)
            )
        return interval

    def _reserve(self) -> float:
        """Claim the next send slot; return how long the caller should wait."""
        with self._lock:
            now = time.monotonic()
            interval = self._interval_locked()
            reserved_at = max(now, self._next_send_at)
            # Advance the queue by the true interval even if this caller's own
            # wait gets clamped below, so clamping never rewinds the schedule.
            self._next_send_at = reserved_at + interval
            self._sends += 1
            wait = reserved_at - now
            if wait > self.config.max_wait:
                self._capped_waits += 1
                wait = self.config.max_wait
            if wait > 0:
                self._waits += 1
                self._total_wait += wait
            return wait

    def acquire(self) -> float:
        """Block until it is this caller's turn to send. Returns seconds waited."""
        wait = self._reserve()
        if wait > 0:
            time.sleep(wait)
        return wait

    async def acquire_async(self) -> float:
        """Async variant of `acquire`. Returns seconds waited."""
        wait = self._reserve()
        if wait > 0:
            await asyncio.sleep(wait)
        return wait

    # ------------------------------------------------------------------ #
    # observation
    # ------------------------------------------------------------------ #

    def _stall_locked(self, seconds: float) -> None:
        if seconds <= 0:
            return
        now = time.monotonic()
        self._next_send_at = max(self._next_send_at, now + seconds)

    def _apply_snapshot_locked(self, snap: RateLimitSnapshot) -> None:
        if not snap.has_rate_limit_info:
            return
        self._seen_headers = True
        request_rate = snap.request_refill_rate
        if request_rate is not None:
            self._request_rate = request_rate
        token_rate = snap.token_refill_rate
        if token_rate is not None:
            self._token_rate = token_rate

        # Budget already below our reserve: stall until it has refilled.
        if (
            snap.remaining_requests is not None
            and snap.remaining_requests <= self.config.min_remaining_requests
        ):
            self._stall_locked(self._refill_wait(snap, tokens=False))
        if (
            snap.remaining_tokens is not None
            and snap.remaining_tokens <= self.config.min_remaining_tokens
        ):
            self._stall_locked(self._refill_wait(snap, tokens=True))

    def _refill_wait(self, snap: RateLimitSnapshot, tokens: bool) -> float:
        """Seconds until the budget is back above our reserve."""
        if tokens:
            remaining = snap.remaining_tokens
            reserve = self.config.min_remaining_tokens
            rate = snap.token_refill_rate or self._token_rate
            reset = snap.reset_tokens
        else:
            remaining = snap.remaining_requests
            reserve = self.config.min_remaining_requests
            rate = snap.request_refill_rate or self._request_rate
            reset = snap.reset_requests
        if remaining is None:
            return 0.0
        deficit = reserve + 1 - remaining
        if deficit <= 0:
            return 0.0
        if rate and rate > 0:
            return min(deficit / rate, self.config.max_wait)
        # No rate known: the provider's own reset estimate is the best we have.
        return min(reset or 0.0, self.config.max_wait)

    def observe_response(
        self,
        headers: Optional[Mapping[str, Any]] = None,
        tokens_used: Optional[int] = None,
    ) -> None:
        """Learn from a successful response: its headers and its token usage.

        Args:
            headers: Response headers, if the provider/transport exposes them.
            tokens_used: Total tokens billed for this request, if reported.
                Used as an EWMA estimate of the per-request token cost, which
                is what makes token-budget pacing possible. It is an estimate,
                not a guarantee of provider quota compliance.
        """
        snap = RateLimitSnapshot.from_headers(headers) if headers is not None else None
        with self._lock:
            if snap is not None:
                self._apply_snapshot_locked(snap)
            if tokens_used is not None and tokens_used > 0:
                if self._avg_tokens is None:
                    self._avg_tokens = float(tokens_used)
                else:
                    self._avg_tokens = 0.7 * self._avg_tokens + 0.3 * tokens_used
            if self._fallback_interval > 0:
                # Recover the send rate on every success.
                decayed = self._fallback_interval * self.config.recovery_factor
                self._fallback_interval = 0.0 if decayed < 1e-3 else decayed

    def observe_rate_limit_error(
        self, headers: Optional[Mapping[str, Any]] = None
    ) -> None:
        """Learn from a 429: back off, and stall until the budget recovers."""
        snap = RateLimitSnapshot.from_headers(headers) if headers is not None else None
        with self._lock:
            self._rate_limit_errors += 1
            if snap is not None and snap.has_rate_limit_info:
                self._apply_snapshot_locked(snap)
                self._stall_locked(
                    min(
                        snap.retry_after or snap.reset_requests or 0.0,
                        self.config.max_wait,
                    )
                )
            else:
                # No usable headers: multiplicatively reduce the send rate.
                self._fallback_interval = min(
                    self.config.max_interval,
                    max(
                        self.config.error_interval,
                        self._fallback_interval * self.config.backoff_factor,
                    ),
                )
                if snap is not None and snap.retry_after:
                    self._stall_locked(min(snap.retry_after, self.config.max_wait))
                else:
                    self._stall_locked(self._fallback_interval)
            logger.debug(
                "rate limiter %s: 429 observed, interval now %.4fs",
                self.name,
                self._interval_locked(),
            )

    # ------------------------------------------------------------------ #
    # introspection
    # ------------------------------------------------------------------ #

    def stats(self) -> Dict[str, Any]:
        """Counters and discovered state, for tests and diagnostics.

        `sends` is the "did it run at all" signal: zero means the limiter was
        never consulted, which is different from "it was consulted and never
        needed to wait" (`sends > 0, waits == 0`).
        """
        with self._lock:
            return dict(
                name=self.name,
                sends=self._sends,
                waits=self._waits,
                total_wait=self._total_wait,
                capped_waits=self._capped_waits,
                rate_limit_errors=self._rate_limit_errors,
                seen_headers=self._seen_headers,
                request_rate=self._request_rate,
                token_rate=self._token_rate,
                avg_tokens=self._avg_tokens,
                fallback_interval=self._fallback_interval,
                interval=self._interval_locked(),
            )


_limiters: Dict[str, RateLimiter] = {}
_registry_lock = threading.Lock()


def get_rate_limiter(key: str, config: Optional[RateLimitConfig] = None) -> RateLimiter:
    """Get (creating if needed) the process-wide limiter for `key`.

    Sharing by key is what lets the cloned agents of a batch job pace against
    one budget. The config of the *first* caller for a given key wins; later
    callers get the existing limiter unchanged.

    Args:
        key: Sharing key, e.g. `"https://api.openai.com/v1::gpt-4o-mini"`.
        config: Settings to use if the limiter does not exist yet.

    Returns:
        The shared `RateLimiter` for `key`.
    """
    with _registry_lock:
        limiter = _limiters.get(key)
        if limiter is None:
            limiter = RateLimiter(config=config, name=key)
            _limiters[key] = limiter
        return limiter


def reset_rate_limiters() -> None:
    """Drop all shared limiters (test/teardown helper)."""
    with _registry_lock:
        _limiters.clear()


def rate_limit_error_headers(exc: BaseException) -> Optional[Mapping[str, Any]]:
    """Classify an exception as a rate-limit error and return its headers.

    Returns:
        None if `exc` is not a rate-limit (429) error; otherwise the response
        headers, which may be an empty mapping if the provider sent none.
        An empty mapping is therefore meaningfully different from None.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    is_rate_limit = status == 429
    if not is_rate_limit:
        # litellm and some SDKs signal rate limits by exception class name
        # without exposing a status code.
        is_rate_limit = type(exc).__name__ in (
            "RateLimitError",
            "APIRateLimitError",
        )
    if not is_rate_limit:
        return None
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        headers = getattr(exc, "headers", None)
    if headers is None:
        return {}
    try:
        return dict(headers)
    except Exception:
        return {}
