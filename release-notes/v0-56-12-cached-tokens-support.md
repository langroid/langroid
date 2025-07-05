# v0.56.12: Cached Tokens Support

## Overview
This release adds support for tracking cached tokens in LLM API responses, enabling accurate cost calculations when using prompt caching features (e.g., OpenAI's prompt caching).

## Key Features

### 1. Cached Token Tracking
- Added `cached_tokens` field to `LLMTokenUsage` class
- Properly extracts cached token counts from OpenAI API responses (both streaming and non-streaming)
- Updated token usage string representation to show cached tokens

### 2. Cost Calculation Updates
- Enhanced cost calculation formula: `(prompt - cached) * input_cost + cached * cached_cost + completion * output_cost`
- Added `cached_cost_per_million` field to `ModelInfo` for all supported models
- Cached token costs typically 25-50% of regular input token costs

### 3. New Model Support
- **Gemini 2.5 Pro**: 1M context, $1.25/$0.31/$10.00 per million tokens
- **Gemini 2.5 Flash**: 1M context, $0.30/$0.075/$2.50 per million tokens
- **Gemini 2.5 Flash Lite Preview**: 64K context, $0.10/$0.025/$0.40 per million tokens

## Code Changes

### Updated Methods
- `compute_token_cost()` in `Agent` class now accepts cached token parameter
- `chat_cost()` returns 3-tuple: (input_cost, cached_cost, output_cost) per 1000 tokens
- `_cost_chat_model()` in OpenAI implementation properly accounts for cached tokens

### API Response Handling
```python
# Cached tokens extracted from OpenAI responses:
cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
```

## Testing
- Added comprehensive tests for cached token tracking
- Verified cost calculations with cached tokens
- All existing tests pass without modification

## Breaking Changes
None - all changes are backward compatible.

## Credits
Original implementation by @alexagr in PR #882, with enhancements in PR #884.