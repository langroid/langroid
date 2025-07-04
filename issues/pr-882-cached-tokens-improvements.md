# PR #882: Cached Tokens Support - Improvements

## Summary
Enhanced PR #882 which adds cached token tracking to LLMTokenUsage. Made several improvements including cleanup of unused code, bug fixes, added tests, and new model support.

## Changes

### 1. Code Cleanup
- Removed unused `chat_cost_per_1k_tokens` and `completion_cost_per_1k_tokens` fields from `LLMConfig` in `base.py`
- These fields were superseded by the ModelInfo approach but were still being updated unnecessarily

### 2. Bug Fixes
- Fixed type error in `openai_gpt.py` when extracting `prompt_tokens_details` from API responses
- Added proper type annotation and type checking to handle cases where the field might not be a dict

### 3. Added Tests
- `test_cached_tokens_tracking()`: Verifies cached tokens are properly tracked in API responses and cost calculations work correctly
- `test_cached_tokens_in_llm_response()`: Tests the LLMTokenUsage class directly including string representation and reset functionality

### 4. Added Gemini 2.5 Model Support
- Fixed `GEMINI_2_5_PRO` enum to map to `"gemini-2.5-pro"` instead of experimental version
- Added new enums: `GEMINI_2_5_FLASH` and `GEMINI_2_5_FLASH_LITE_PREVIEW`
- Added complete ModelInfo entries with proper costs and parameters:
  - **Gemini 2.5 Pro**: 1M context, $1.25/$0.31/$10.00 per million tokens
  - **Gemini 2.5 Flash**: 1M context, $0.30/$0.075/$2.50 per million tokens  
  - **Gemini 2.5 Flash Lite Preview**: 64K context, $0.10/$0.025/$0.40 per million tokens

## Testing
- All existing tests pass
- New tests verify cached token functionality
- Code passes all linting and type checking