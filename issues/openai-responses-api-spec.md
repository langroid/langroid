# Langroid: OpenAI Responses API Support — Architecture & Design Spec

This document specifies the architecture, design, and implementation plan for adding first‑class support for the OpenAI Responses API to Langroid. The new module will live alongside the existing OpenAI Chat Completions integration in `langroid/language_models/openai_gpt.py`, and will be exposed as a new provider implementing Langroid’s `LanguageModel` interface.

Owner: Langroid core
Status: Draft (implementation-ready)
Scope: Add new Responses-based provider + config, mapping, streaming, tools, structured outputs, multimodal, usage/cost, caching, retries, and examples/docs.


## Goals

- Add a Langroid LLM provider that uses the OpenAI Responses API for chat, tool calling, structured outputs, reasoning stream, and multimodal inputs.
- Keep full compatibility with the existing `LanguageModel` interfaces used by agents (messages, tools, usage, cost, streaming APIs).
- Provide robust streaming and non-streaming paths with retries and cache integration.
- Preserve existing agent behavior; adoption should require only a config swap.


## Non-Goals

- Support for non-OpenAI “OpenAI-like” backends via Responses (Groq, Cerebras, etc.) on day one.
- Azure Responses parity (can be implemented later as a sibling module).
- Rewriting agent prompts or higher-level chat logic; this is a drop-in LLM provider.


## High-Level Design

- New module: `langroid/language_models/openai_responses.py`.
- New classes:
  - `OpenAIResponsesConfig`: configuration for Responses API (extends `OpenAIGPTConfig` to reuse auth, headers, http client, retry, cache, cost knobs; ignores irrelevant fields).
  - `OpenAIResponses`: concrete provider implementing `LanguageModel` using `client.responses.create` and `client.responses.stream`.
- `LanguageModel.create` will route `config.type == "openai_responses"` to `OpenAIResponses`.
- Export from `langroid/language_models/__init__.py` for public availability.


## Configuration

Base class choice: extend `OpenAIGPTConfig` to maximize reuse (auth, proxies, headers, http client factory/config, retry params, cache settings, cost helpers). Fields not applicable to Responses are ignored gracefully.

Key fields:
- `type`: `"openai_responses"` (used by `LanguageModel.create`).
- `api_key`, `api_base`, `headers`, `http_client_factory`, `http_client_config`, `http_verify_ssl`: reuse.
- `chat_model`: OpenAI model name (e.g., `gpt-4.1`, `gpt-4o`, `o3-mini`, `o1-mini`).
- `temperature`, `top_p`, `seed`, `timeout`, `stream`, `retry_params`: reuse.
- `max_output_tokens`: mapped to Responses `max_output_tokens` (note: not `max_tokens`).
- `params`: extend existing `OpenAICallParams` to include `reasoning_effort` and other Responses-relevant knobs; mapped to `reasoning={"effort": ...}` for o‑models.
- `supports_json_schema`, `supports_strict_tools`: default from `model_info` when `api_base is None` (OpenAI endpoints); conservative `False` otherwise.
- `parallel_tool_calls`: pass-through if supported by Responses.
- `cache_config`, `use_cached_client`: reuse.
- Optional `beta_headers` toggle if certain Responses features require `OpenAI-Beta`.

Validation:
- If `chat_model` is not an OpenAI Responses-capable model (per `model_info`), raise a clear error.
- Filter/rename params for quirks (e.g., o‑series) similar to `openai_gpt.py`’s `unsupported_params` and `rename_params`.


## Client and Transport

- Use `OpenAI` / `AsyncOpenAI` via `get_openai_client` / `get_async_openai_client` (from `client_cache.py`).
- Respect `api_base`, `headers`, and injected HTTP client per the three-tier strategy already used by `openai_gpt.py`:
  1) `http_client_factory` (not cacheable),
  2) `http_client_config` (cacheable),
  3) `http_verify_ssl=False` (cacheable, simple bypass).
- Only support OpenAI base at first; if `chat_model` implies non-OpenAI provider, raise a helpful error.


## Request Mapping (LLMMessage → Responses input)

The Responses API unifies chat, tools, and multimodal into a single `input` (and optional `instructions`). We convert Langroid’s `List[LLMMessage]` to Responses as follows:

Roles:
- `Role.SYSTEM`: join all system messages into a top-level `instructions` string (joined with double newlines). Optionally also attach system messages in `input` with `role: system` if needed later.
- `Role.USER`: map to content parts with `{type: "input_text", text: ...}` and `{type: "input_image", ...}` (for attachments).
- `Role.ASSISTANT`: prior assistant text → `{type: "output_text", text: ...}`; prior tool calls are not typically replayed unless intentionally reconstructing state.
- `Role.TOOL` / `Role.FUNCTION`: results of tool executions → `{type: "tool_result", tool_call_id: ..., output: <string>, is_error?: bool}`.

Attachments (via `FileAttachment`):
- `image_url`: wrap into `{type: "input_image", image_url: {url, detail?}}`.
- `file` (data URI): for images/PDFs, use `{type: "input_image", image_url: {url: data:...}}`. Start with images and PDFs; other file types can be iterated later if Responses adds `input_file`.

Tools and tool choice:
- `tools: List[OpenAIToolSpec]` pass-through; already OpenAI-compatible. Include `strict` if supported.
- `tool_choice`: support `"auto" | "none" | "required"` or `{"type": "function", "function": {"name": "..."}}`.

Structured outputs:
- `response_format: OpenAIJsonSchemaSpec` → pass `{type: "json_schema", json_schema: {name, description, schema, strict?}}` directly.
- Optionally support `{type: "json_object"}` when schema is not provided.

Parameters:
- `model`, `max_output_tokens` (derived from `config.model_max_output_tokens` and context), `temperature`, `top_p`, `seed`, `user`.
- `reasoning`: for o‑series, `{"effort": config.params.reasoning_effort}`.
- `parallel_tool_calls`: include if enabled.


## Response Mapping (Responses → LLMResponse)

Non-stream path (`client.responses.create`):
- The Responses object contains `output` entries such as `message`, `tool_call`, and `tool_result`.
- Aggregate assistant text by concatenating all `text` from `message.content` parts.
- Collect `reasoning` text if present (o‑models) from `message.content` reasoning parts.
- Convert `tool_call` entries into `List[OpenAIToolCall]` with `function.name` and parsed `function.arguments` (via `parse_imperfect_json`). Preserve `id`.
- Usage: read `usage.input_tokens`, `usage.output_tokens`, and `usage.input_tokens_details.cached_tokens` if available.
- Return `LLMResponse` with `message`, `reasoning`, `oai_tool_calls`, and `usage`.

Stream path (`client.responses.stream`):
- Handle events:
  - `response.output_text.delta`: append `delta` to completion; stream via `config.streamer(..., StreamEventType.TEXT)`.
  - `response.reasoning.delta`: append to reasoning; stream as TEXT (same mechanism; the CLI may style it dimmer but API-wise it’s the same event type).
  - `response.tool_call.delta`: accumulate deltas per tool index (`id`, `type`, and concatenated `function.arguments` string). Stream name/args via `TOOL_NAME` and `TOOL_ARGS` events.
  - `response.completed`: capture `final_response` for usage and caching; end loop.
  - `response.error`: raise to trigger retry.
- After stream completes, convert tool deltas into `OpenAIToolCall` using logic analogous to `OpenAIGPT.tool_deltas_to_tools`, parsing args with `parse_imperfect_json` and omitting malformed tool calls (moving their raw content into `message` as a last resort when necessary).
- Cache the `final_response.model_dump()` snapshot post-stream (see Caching section).


## Unsupported and Renamed Params

- Maintain `rename_params()` and `unsupported_params()` akin to `openai_gpt.py`, driven by `model_info` and special-casing notable model families (e.g., o‑series). Filter out unsupported keys from the final request to avoid 400s.


## Caching

- Reuse Redis-based cache via `cache_config` and helpers (`_cache_lookup`, `_cache_store`) as in `openai_gpt.py`.
- Cache key: SHA256 hash of a canonical request dict: `model + instructions + input payload + tools + tool_choice + response_format + params`.
- Non-stream path: store the `response.model_dump()`.
- Stream path: do not cache the generator; after `response.completed`, store the reconstructed `final_response.model_dump()`.
- Respect global `settings.cache` on/off.


## Retries, Timeouts, Errors

- Wrap core calls with `retry_with_exponential_backoff` / `async_retry_with_exponential_backoff` using `config.retry_params`.
- For streaming, attempt to read the first event to detect immediate errors (rate-limit/auth) and rebuild the stream on success, mirroring `openai_gpt.py`.
- Respect `timeout` via underlying OpenAI client options.
- Use `friendly_error` to surface clear, user-centric messages.


## Usage and Cost Accounting

- Compute costs using `LanguageModel.chat_cost()` and `model_info` rates for the active model.
- Token counts:
  - Prompt tokens = `usage.input_tokens - (usage.input_tokens_details.cached_tokens or 0)`.
  - Cached tokens = `usage.input_tokens_details.cached_tokens` (when present).
  - Completion tokens = `usage.output_tokens`.
- Update `usage_cost_dict` accordingly and include `LLMTokenUsage` in `LLMResponse.usage`.


## Compatibility and Integration

- `LanguageModel.create`: add case to return `OpenAIResponses` when `config.type == "openai_responses"`.
- `supports_functions_or_tools()`: use `model_info.has_tools` for the current model.
- `supports_json_schema`: `True` when `api_base is None` (OpenAI) and `model_info.has_structured_output`.
- No changes to agent APIs; `LLMMessage` and `LLMResponse` remain the lingua franca.


## Testing Plan

Unit (no network):
- Message conversion:
  - System → `instructions` (concatenation verified).
  - User text and images/files → `input_text` and `input_image` parts.
  - Assistant text → `output_text` parts.
  - Tool results → `tool_result` with correct `tool_call_id`.
- Tools and strict mode: `OpenAIToolSpec` → API dict; `tool_choice` mapping.
- Structured output mapping: `OpenAIJsonSchemaSpec` → `response_format`.
- Streaming aggregation:
  - Aggregate output text from `response.output_text.delta`.
  - Accumulate tool_call deltas and parse into `OpenAIToolCall`.
  - Reasoning deltas collected into `LLMResponse.reasoning`.
- Caching:
  - Non-stream: identical requests hit cache.
  - Stream: cache populated after `completed`.

Mocked client:
- Simulate `responses.create` returning a composed `output` list with `message`, `tool_call`, and `usage` fields.
- Simulate streaming sequence (deltas → completed) and error path.

Integration (requires `OPENAI_API_KEY`, marked slow):
- Basic chat.
- Tools: single simple tool; verify tool_call and tool_result round-trip.
- Structured output with a small JSON schema.
- Reasoning model (`o3-mini` or `o1-mini`) with `reasoning_effort`; assert reasoning text present.
- Vision: image input with `gpt-4o`.
- Usage/cost: token counts present; calls tracked.


## Documentation & Examples

New examples:
- `examples/openai_responses/tools_json_structured.py`: tools + schema + streaming.
- `examples/openai_responses/vision.py`: image input with a short output; streaming enabled.
- `examples/openai_responses/reasoning.py`: o‑series reasoning with streamed deltas and `reasoning_effort`.

Docs:
- README section: Choosing Responses vs Chat Completions (when to use which).
- Config snippet:

```python
from langroid.language_models.openai_responses import OpenAIResponsesConfig
from langroid.language_models.base import LLMMessage, Role

cfg = OpenAIResponsesConfig(
    chat_model="gpt-4.1",
    stream=True,
)

llm = LanguageModel.create(cfg)
resp = llm.chat([
    LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
    LLMMessage(role=Role.USER, content="Summarize this in 3 bullets:"),
])
print(resp.message)
```

- Tool calling and structured output examples; reasoning model note and `reasoning_effort`.


## Migration Notes

- Safe to run side-by-side with `OpenAIGPT`.
- Prefer `OpenAIResponses` for unified tool+multimodal and reasoning streams.
- Existing agents do not change; only swap `config.type` to `"openai_responses"` and set `chat_model` to a Responses-capable model.


## Open Questions / Iteration Targets

- Non-image file inputs (e.g., non-PDF docs): start with images and PDFs via data URIs; expand if/when OpenAI exposes an `input_file` pathway broadly.
- Azure Responses parity: likely a follow-up module mirroring `azure_openai.py`.
- Whether to add a global flag to route `OpenAIGPT` through Responses internally: not recommended initially to avoid surprises.


## Implementation Outline (Pseudocode)

File: `langroid/language_models/openai_responses.py`

```python
class OpenAIResponsesConfig(OpenAIGPTConfig):
    type: str = "openai_responses"
    # inherits api_key, api_base, headers, retry, cache, http client knobs, etc.
    # ensure irrelevant fields (use_chat_for_completion, completion_model, formatter) are ignored

class OpenAIResponses(LanguageModel):
    client: OpenAI | None
    async_client: AsyncOpenAI | None

    def __init__(self, config: OpenAIResponsesConfig = OpenAIResponsesConfig()):
        cfg = config.model_copy()
        super().__init__(cfg)
        # set up clients via client_cache with http client injection like openai_gpt.py
        # set supports_json_schema/strict_tools from model_info when api_base is None

    def set_stream(self, stream: bool) -> bool: ...
    def get_stream(self) -> bool: ...

    def generate(self, prompt: str, max_tokens: int = 200) -> LLMResponse:
        # Wrap into a minimal Responses request using `input = [{type: "input_text", text: prompt}]`
        # Call create(); map result to LLMResponse

    async def agenerate(...): ...

    def chat(self, messages, max_tokens=200, tools=None, tool_choice="auto", functions=None, function_call="auto", response_format=None) -> LLMResponse:
        # Convert messages -> instructions + input parts
        # Build request including tools, tool_choice, response_format, params
        # Call create() or stream(); handle retries, caching, usage

    async def achat(...): ...

    # Helpers: conversion, streaming event processing, tool delta accumulation, cache lookup/store,
    # cost/usage computation (reuse chat_cost and LLMTokenUsage)
```

Key helper functions:
- `messages_to_responses_input(messages) -> (instructions: str, input_parts: list)`
- `attachments_to_input_parts(files, model) -> list`
- `tool_deltas_to_calls(deltas) -> (raw_content: str, tool_calls: List[OpenAIToolCall])`
- `_process_stream_event(event, ...) -> state transitions` (sync and async versions)


## Acceptance Criteria

- New provider compiles and integrates with `LanguageModel.create`.
- Non-stream and stream chats work with text-only, tools, and JSON schema.
- Vision example works with image input for `gpt-4o`.
- Reasoning example works and captures streamed reasoning.
- Usage and cost metrics are recorded and exposed via `LanguageModel.usage_cost_summary()`.
- Caching works for non-streaming calls and stores post-stream snapshots for streaming.
- Documentation and examples are added and pass basic sanity checks.

