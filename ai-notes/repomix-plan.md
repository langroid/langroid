## Plan to Add llms-no-tests.txt

### Overview
Create a third version of the repomix output that excludes all test files from the `tests/` directory. This will provide a more concise version focused only on source code without test implementations.

### Steps:

1. **Create ai-scratchpads directory and save this plan** ✓
   - Create directory: `mkdir -p ai-scratchpads`
   - Save this plan to `ai-scratchpads/repomix-plan.md`

2. **Create temporary repomix configuration**
   - Copy existing `repomix.config.json` to `repomix-no-tests.config.json`
   - Add `"tests/**"` to the `customPatterns` array in the `ignore` section
   - Add `"llms-no-tests.txt"` to the ignore patterns to prevent recursive inclusion

3. **Generate the new output file**
   - Run: `repomix --config repomix-no-tests.config.json -o llms-no-tests.txt`
   - This will create a new file excluding all test files

4. **Clean up and update documentation**
   - Remove the temporary `repomix-no-tests.config.json` file
   - Update `ai-instructions/claude-repomix-instructions.md` to mention the third variant
   - Add a note about generating the no-tests version with the command:
     ```bash
     # No-tests version (excludes tests directory)
     repomix --config repomix-no-tests.config.json -o llms-no-tests.txt
     ```

### Expected Result
- A new file `llms-no-tests.txt` that contains all source code except test files
- This will be smaller than the standard `llms.txt` but larger than `llms-compressed.txt`
- Useful for LLM analysis when test implementations are not needed

### File Size Expectations
Based on the current setup:
- `llms.txt`: ~3.3 MB (782K tokens)
- `llms-compressed.txt`: ~1.6 MB (434K tokens)
- `llms-no-tests.txt`: Expected to be between these sizes, excluding test code

## Results and Conclusions

### Actual Token Counts
After generating all variants, here are the actual token counts:
- `llms.txt`: 782K tokens (standard version with tests)
- `llms-compressed.txt`: 434K tokens (compressed version with tests)
- `llms-no-tests.txt`: 652K tokens (no tests version)
- `llms-no-tests-compressed.txt`: 400K tokens (compressed no-tests version)

### Key Observations
1. **Limited Impact of Excluding Tests**: Removing test files only reduced tokens by ~130K (17% reduction), suggesting that test files don't constitute a major portion of the codebase.

2. **Compression More Effective**: The compression feature provides a much more significant reduction (~45-50% reduction) compared to just excluding tests.

3. **Minimal Benefit of Combined Approach**: The compressed no-tests version (400K) is only marginally smaller than the compressed version with tests (434K) - a difference of just 34K tokens or ~8%.

### Recommendations
- For most use cases, the standard `llms-compressed.txt` (434K tokens) is likely sufficient
- The no-tests variants might be useful for specific scenarios where test implementation details would confuse the LLM or are explicitly not needed
- The marginal benefit of excluding tests doesn't justify maintaining multiple variants unless there's a specific need

### Files Created
- `repomix-no-tests.config.json` - Permanent config file for generating no-tests versions
- `llms-no-tests.txt` - Full version without tests (652K tokens)
- `llms-no-tests-compressed.txt` - Compressed version without tests (400K tokens)