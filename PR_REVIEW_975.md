# PR #975 Review: Remove traceback from OpenAI API error logs

**Author:** alexagr
**Branch:** `api_error_log` → `main`
**Changed file:** `langroid/language_models/openai_gpt.py` (+12, -0)

## Summary

This PR adds `except openai.APIError` handlers before the generic `except Exception`
blocks in the four public methods of `OpenAIGPT`: `generate`, `agenerate`, `chat`,
and `achat`. The intent is to log API errors cleanly (without a full traceback) since
server-side errors don't benefit from a local stack trace.

The motivation is sound — `friendly_error()` includes `traceback.format_exc()` which
produces multi-line stack traces for every OpenAI API error. For authentication
failures, bad requests, and similar server-side errors these tracebacks are noisy and
provide no diagnostic value.

## Issues

### 1. `openai.APIError` is too broad — catches connection and timeout errors too

`openai.APIError` is the base class for the entire OpenAI exception hierarchy:

```
openai.APIError
├── openai.APIConnectionError    ← network/local issues
│   └── openai.APITimeoutError   ← timeout issues
└── openai.APIStatusError        ← HTTP status errors from the API server
    ├── openai.BadRequestError (400)
    ├── openai.AuthenticationError (401)
    ├── openai.PermissionDeniedError (403)
    ├── openai.NotFoundError (404)
    ├── openai.UnprocessableEntityError (422)
    ├── openai.RateLimitError (429)
    └── openai.InternalServerError (>=500)
```

The PR description correctly identifies that server-side errors (AuthenticationError,
BadRequestError, etc.) don't benefit from tracebacks. However, `APIConnectionError`
and `APITimeoutError` **are** related to the local environment (network configuration,
DNS, proxy issues), where a traceback **could** help diagnose the problem.

**Recommendation:** Use `openai.APIStatusError` instead of `openai.APIError`. This
captures exactly the server-side HTTP errors (400, 401, 403, 404, 422, 429, 500+)
while letting connection/timeout errors fall through to the `except Exception` handler
where `friendly_error()` provides the full traceback.

### 2. `raise e` vs bare `raise`

The PR uses `raise e` which resets the exception's `__traceback__` attribute. A bare
`raise` preserves the original traceback chain. This is consistent with the existing
code (the `except Exception` blocks also use `raise e`), but `raise` is generally
preferred — especially since callers higher up the stack may want the full traceback
context even if the log message omits it.

This is a minor style point and not a blocker — it could be a separate cleanup across
the file.

### 3. Log level consideration

Using `logging.error()` is appropriate for most API errors, but for `RateLimitError`
(a subclass of `APIStatusError`) `logging.warning()` might be more fitting since it's
a transient condition. That said, by the time the error reaches this outer handler the
retry logic in `utils.py` has already been exhausted, so `error` level is reasonable.

Not a blocker.

## Suggested Change

Replace `openai.APIError` with `openai.APIStatusError` in all four handlers:

```python
except openai.APIStatusError as e:
    logging.error(f"API error in OpenAIGPT.generate: {e}")
    raise e
```

## Verdict

The change addresses a real usability issue — excessively noisy tracebacks for
server-side API errors. With the suggested narrowing from `APIError` to
`APIStatusError`, this would be a clean, well-targeted improvement. The code is
consistent with existing patterns and correctly placed before the generic exception
handlers.

**Recommendation: Request changes** — use `openai.APIStatusError` instead of
`openai.APIError` to avoid suppressing tracebacks for connection/timeout errors.
