# New Model Support: GPT-5.1 and Gemini 3.0

## Objective

Add support for newly released models to Langroid's `model_info.py`:

- GPT-5.1 variants (gpt-5.1, gpt-5.1-chat, gpt-5.1-codex, gpt-5.1-codex-mini)
- Gemini 3.0 variants (to be determined from models.dev)

## Background

New models have been released by OpenAI and Google that need to be added to
Langroid's model registry. This ensures users can leverage these models with
proper cost tracking, context length limits, and feature support.

## Information Sources

- Model specs (context length, costs): https://models.dev/
- OpenAI feature support: https://platform.openai.com/docs/api-reference/chat
- Assumption: GPT-5.1 features similar to GPT-5

## GPT-5.1 Model Information

Based on models.dev data (as of Nov 2025):

### 1. gpt-5.1
- **Context Length**: 272,000 tokens
- **Max Output**: 128,000 tokens
- **Input Cost**: $1.25 per 1M tokens
- **Output Cost**: $10.00 per 1M tokens
- **Cache Read Cost**: $0.13 per 1M tokens
- **Notes**: Released 2024-09, Azure variant

### 2. gpt-5.1-chat
- **Context Length**: 128,000 tokens
- **Max Output**: 16,384 tokens
- **Input Cost**: $1.25 per 1M tokens
- **Output Cost**: $10.00 per 1M tokens
- **Cache Read Cost**: $0.13 per 1M tokens
- **Notes**: Released 2024-09, Azure variant

### 3. gpt-5.1-codex
- **Context Length**: 400,000 tokens
- **Max Output**: 128,000 tokens
- **Input Cost**: $1.25 per 1M tokens
- **Output Cost**: $10.00 per 1M tokens
- **Cache Read Cost**: $0.13 per 1M tokens
- **Notes**: Released 2024-09, Azure variant, code-optimized

### 4. gpt-5.1-codex-mini
- **Context Length**: 400,000 tokens
- **Max Output**: 128,000 tokens
- **Input Cost**: $0.25 per 1M tokens
- **Output Cost**: $2.00 per 1M tokens
- **Cache Read Cost**: $0.03 per 1M tokens
- **Notes**: Released 2024-09, Azure variant, code-optimized, cheaper

## GPT-5.1 Feature Support

Based on similarity to GPT-5 (to be confirmed from OpenAI API reference):

- **has_tools**: `False` (reasoning models typically don't support tools)
- **has_structured_output**: `True` (likely similar to GPT-5)
- **allows_streaming**: `True` (default)
- **allows_system_message**: `True` (default)
- **unsupported_params**: `["temperature"]` (likely similar to GPT-5)
- **rename_params**: `{"max_tokens": "max_completion_tokens"}` (likely)
- **Special parameters**: May support `reasoning_effort` (to be confirmed)

## Gemini 3.0 Model Information

**TO BE DETERMINED**: Need to fetch from models.dev

Expected variants based on previous patterns:
- gemini-3.0-pro
- gemini-3.0-flash
- gemini-3.0-flash-lite

Information needed for each:
- Context length
- Max output tokens
- Input/output costs
- Cached input costs
- Feature support flags

## Implementation Tasks

### 1. Add Enum Entries

In `langroid/language_models/model_info.py`:

**OpenAIChatModel enum** (add after existing GPT-5 models):
```python
class OpenAIChatModel(ModelName):
    # ... existing models ...
    GPT5_1 = "gpt-5.1"
    GPT5_1_CHAT = "gpt-5.1-chat"
    GPT5_1_CODEX = "gpt-5.1-codex"
    GPT5_1_CODEX_MINI = "gpt-5.1-codex-mini"
```

**GeminiModel enum** (add after existing Gemini 2.5 models):
```python
class GeminiModel(ModelName):
    # ... existing models ...
    GEMINI_3_0_PRO = "gemini-3.0-pro"  # if exists
    GEMINI_3_0_FLASH = "gemini-3.0-flash"  # if exists
    GEMINI_3_0_FLASH_LITE = "gemini-3.0-flash-lite"  # if exists
```

### 2. Add MODEL_INFO Entries

Add comprehensive `ModelInfo` entries for each new model with:
- Provider (OpenAI or Google)
- Context length
- Max output tokens
- Costs (input, output, cached)
- Feature flags
- API parameter quirks
- Description

### 3. Update OpenAI_API_ParamInfo (if needed)

If GPT-5.1 supports `reasoning_effort` or other special parameters, add to
the appropriate parameter lists.

### 4. Verification

After implementation:
- Run `make check` to ensure linting and type checking pass
- Verify model names are accessible via the enums
- Verify costs and limits are correctly set
- Check that feature flags match OpenAI API capabilities

## Questions/Clarifications Needed

1. **Gemini 3.0**: Does this model exist yet? If so, what are the exact variant
   names and specs?

2. **GPT-5.1 Feature Support**: Should we confirm all feature flags from the
   OpenAI API reference, or is assuming similarity to GPT-5 acceptable?

3. **Special Parameters**: Do GPT-5.1 models support `reasoning_effort` or
   other special parameters?

4. **Provider**: The models.dev data shows these as "Azure" variants - should
   they still use `ModelProvider.OPENAI`?

## Files to Modify

- `langroid/language_models/model_info.py`
  - Add enum entries for new models
  - Add MODEL_INFO dictionary entries
  - Update OpenAI_API_ParamInfo if needed

## Testing

No specific unit tests are required for individual model definitions (per user
guidance). The implementation focuses on:
- Correct model name registration
- Accurate API cost tracking
- Proper context length limits
- Correct feature support flags

## References

- models.dev: https://models.dev/
- OpenAI Chat API: https://platform.openai.com/docs/api-reference/chat
- Existing GPT-5 implementation: `langroid/language_models/model_info.py:323-364`
- Existing Gemini 2.5 implementation: Similar location in same file
