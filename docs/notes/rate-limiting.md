# Pro-active rate limiting

Langroid's LLM calls have always had reactive retry-with-exponential-backoff:
when the provider answers `429 Too Many Requests`, the call sleeps and retries.
That works, but it is blind — on a large batch job (e.g.
[`run_batch_tasks`](batch-processing.md)) a lot of wall-clock time goes into
hitting the limit and backing off.

The `rate_limit` option adds a **pro-active** layer: requests are *paced* so
the provider's limit is approached rather than exceeded. It is **off by
default**; with it disabled the request path is exactly as it was, and the
retry/backoff logic is untouched either way (it remains the fallback).

## Enabling it

```python
import langroid.language_models as lm

llm_config = lm.OpenAIGPTConfig(
    chat_model="gpt-4o-mini",
    rate_limit=lm.RateLimitConfig(enabled=True),
)
```

Every field can also be set from the environment, with the
`LANGROID_RATE_LIMIT_` prefix:

```bash
export LANGROID_RATE_LIMIT_ENABLED=1
export LANGROID_RATE_LIMIT_HEADROOM=0.2
```

## You do not configure your rate limits

Deliberately, there is no "requests per minute" setting. Limits vary by
account, model and tier, and change without notice, so a configurable budget
just moves the maintenance onto you.

Instead the limiter reads what OpenAI (and most OpenAI-compatible gateways)
report on *every* response:

```
x-ratelimit-limit-requests / x-ratelimit-remaining-requests
x-ratelimit-limit-tokens   / x-ratelimit-remaining-tokens
x-ratelimit-reset-requests / x-ratelimit-reset-tokens
```

`reset` is the time until the bucket refills to `limit`, so a single response
is enough to recover the account's actual sustained limit:

```
rate = (limit - remaining) / reset
```

The limiter then paces sends at `rate * (1 - headroom)`. To keep these headers,
the two chat-completion call sites switch to the OpenAI SDK's
`with_raw_response.create(...)` form when the limiter is enabled — the ordinary
parsed-body form discards headers. Streaming is unaffected.

Token budgets are paced the same way, using an exponentially-weighted average
of the `usage.total_tokens` actually billed per request as the per-request
estimate. That is an **estimate**, not a guarantee of provider quota
compliance. For streaming responses the usage arrives in a trailing chunk
rather than on the response object, so it is picked up as the chunks flow past
the consumer — i.e. it informs the *next* request.

## Providers that report nothing

Local servers, Groq/Cerebras and litellm generally send no rate-limit headers.
There the limiter needs no prior knowledge of any limit either: it reduces the
send rate multiplicatively after a `429` and recovers it on every success
(AIMD on the inter-send interval).

The same applies to a provider that sends *some* of the headers but not enough
to derive a rate from — a `remaining` with no `limit` or `reset`, say. Having
seen a header is not the test; having a rate to pace to is.

## Batch jobs share one limiter

Limiters are shared process-wide, keyed by `(api_base, chat_model)`, so the
many cloned agents of a `run_batch_tasks` run pace against one budget rather
than each discovering the limit separately. Set `share_key` to override the
key — e.g. to make several models on one account share a single budget.

Cache hits never consult the limiter: pacing happens only on the path that
actually calls the API.

## Settings

| field | default | meaning |
|---|---|---|
| `enabled` | `False` | master switch; `False` leaves the request path as-is |
| `headroom` | `0.1` | fraction of the discovered budget left unused |
| `max_wait` | `60.0` | safety valve: never sleep longer than this for one request |
| `warmup_interval` | `0.05` | interval used before the first header is seen |
| `min_remaining_requests` | `1` | stall below this many requests remaining |
| `min_remaining_tokens` | `0` | stall below this many tokens remaining |
| `backoff_factor` | `2.0` | header-free mode: interval multiplier on a `429` |
| `recovery_factor` | `0.9` | header-free mode: interval multiplier on a success |
| `error_interval` | `0.05` | header-free mode: interval floor once a `429` is seen |
| `max_interval` | `10.0` | header-free mode: interval ceiling |
| `share_key` | `None` | override the process-wide sharing key |

## Limitations

- Single process only. Two processes against the same account each discover the
  limit independently, so they will together overshoot it.
- Until the first response arrives there are no headers to learn from, so the
  opening burst of a concurrent batch is paced only by `warmup_interval`.
- A send slot is reserved before the limiter knows what the response will say.
  A `429`, or a budget the provider reports as nearly exhausted, raises a
  cooldown gate that every caller re-checks on waking, so those do reach
  callers that are already queued. A merely *slower* newly-discovered rate
  applies from the next reservation onwards, so a wave already in the queue can
  still drain at the previous pace.
- Token pacing rests on an average of past requests; a sudden jump in prompt
  size can still overshoot the token budget, which is what the retry/backoff
  fallback is for.
- If `max_wait` binds, the request is sent anyway rather than held further, and
  any resulting `429` is handled reactively.

## Diagnostics

`RateLimiter.stats()` reports both whether the limiter ran and what it found:

```python
from langroid.language_models.rate_limiter import get_rate_limiter

stats = get_rate_limiter("default::gpt-4o-mini").stats()
# {'sends': 240, 'waits': 239, 'total_wait': 13.2, 'rate_limit_errors': 0,
#  'seen_headers': True, 'request_rate': 8.33, 'token_rate': 3333.3, ...}
```

`sends == 0` means the limiter was never consulted — which is different from
"it was consulted and never needed to wait" (`sends > 0, waits == 0`).
