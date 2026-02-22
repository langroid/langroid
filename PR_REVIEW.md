# Review: PR #985 — Preserve extra_content in tool call in streaming mode

## Summary

This PR fixes a bug where the `extra_content` field on `OpenAIToolCall` objects is lost during streaming mode. The field is already properly handled in the non-streaming path (via `OpenAIToolCall.from_dict()`), but the streaming path in `tool_deltas_to_tools()` was not preserving it.

**Author:** alexagr
**File changed:** `langroid/language_models/openai_gpt.py` (+7 lines)

## Analysis

### What the PR does

The change touches three places in `OpenAIGPT.tool_deltas_to_tools()`:

1. **Initializes `extra_content`** in the `defaultdict` template (line 1433) — ensures new tool accumulator dicts have the field present as `None`.

2. **Accumulates `extra_content`** from streaming deltas (lines 1452-1456) — when a tool_delta contains a non-None `extra_content`, it's stored on the accumulated tool dict. Uses `.get()` safely so deltas without the key don't raise `KeyError`.

3. **Passes `extra_content`** to the `OpenAIToolCall` constructor (line 1489) — uses `.get()` so it defaults to `None` if absent, matching the model's default.

### Correctness

The changes are correct and consistent with how `extra_content` is handled elsewhere:

- **Non-streaming path** (`openai_gpt.py:2336`): Uses `OpenAIToolCall.from_dict()` which calls `message.get("extra_content")` — same pattern.
- **Serialization** (`base.py:339-340`): Already strips `extra_content` from dicts when it's `None` before sending to the API, so the new `None` default in the accumulator won't leak into API calls.
- **Model definition** (`base.py:176`): `extra_content: Dict[str, Any] | None = None` — the PR's usage is type-consistent.

### Edge cases handled

- **Deltas without `extra_content` key**: The `tool_delta.get("extra_content")` call (not `tool_delta["extra_content"]`) correctly handles deltas from providers that don't include this field at all.
- **`extra_content` is `None`**: The `is not None` check avoids overwriting a previously set value with `None`.
- **No `extra_content` in accumulated dict**: The `tool_dict.get("extra_content")` in the constructor call safely returns `None`.

## Issues

### Minor: Missing trailing comma (style nit)

```python
"type": None,
"extra_content": None   # <-- missing trailing comma
```

The existing entries in the dict literal all use trailing commas. For consistency (and to make future diffs cleaner), this should be:

```python
"extra_content": None,
```

This is cosmetic only — Python accepts it either way.

### Consideration: `extra_content` merging across deltas

The current implementation overwrites `extra_content` on each delta that has it. If a provider ever sends `extra_content` across multiple deltas for the same tool index (with partial content each time), the earlier values would be lost. However, this matches the pattern used for `id` and `type` in the same function (which also overwrite rather than merge), so it's consistent. The `extra_content` field is a `Dict[str, Any]`, not a string that could be concatenated, so overwrite semantics are the right choice.

## Verdict

**Approve with nit.** The fix is correct, minimal, and well-scoped. It properly mirrors the non-streaming path and is consistent with the existing accumulation patterns. The only suggestion is adding a trailing comma for style consistency.
