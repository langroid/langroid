# v0.56.6: DocChatAgent Retrieval Configuration Refactor and Critical Fixes

## Summary
- Refactored retrieval parameter configuration in DocChatAgent for better clarity and control
- Fixed critical passages accumulation logic that could include incorrect documents in results
- Fixed reciprocal rank fusion (RRF) bias that unfairly penalized documents found by only one retrieval method
- Added intelligent configuration validation to prevent invalid retrieval setups
- Maintained backward compatibility with deprecated `n_similar_docs` parameter

## Changes Made

### 1. Moved Retrieval Parameters to Proper Location
- Added `n_relevant_chunks` and `n_similar_chunks` to `DocChatAgentConfig` where they logically belong
- Deprecated `n_similar_docs` in `ParsingConfig` (set to `None` by default)
- These parameters provide clearer semantics:
  - `n_similar_chunks`: number of chunks to retrieve by each method (semantic, BM25, fuzzy)
  - `n_relevant_chunks`: final number of chunks to return after all reranking

### 2. Backward Compatibility
- If users still set the deprecated `n_similar_docs` parameter, it will be used for both new parameters
- A deprecation warning is logged to encourage migration to the new parameters
- This ensures existing code continues to work while encouraging adoption of the new, clearer parameters

### 3. Added Smart Configuration Validation
The DocChatAgent initialization now includes intelligent validation to prevent invalid configurations:

#### Cross-Encoder and RRF Conflict Detection
- If both `cross_encoder_reranking_model` and `use_reciprocal_rank_fusion` are set, warns that RRF will be ignored
- Cross-encoder reranking takes precedence over RRF when both are configured

#### Automatic RRF Enablement
- Automatically enables RRF when all of the following conditions are met:
  - No cross-encoder reranking model is set
  - RRF is currently disabled
  - BM25 or fuzzy matching is enabled
  - `n_relevant_chunks` < `n_similar_chunks` × (number of retrieval methods)
- This prevents situations where multiple retrieval methods are used but there's no way to properly combine their results

### 4. Fixed Critical Passages Accumulation Logic
- Previously had a critical flaw where passages accumulation was inconsistent:
  - When using cross-encoder reranking, BM25 and fuzzy match results were appended to passages
  - But deduplication used `[id2doc[id] for id in id2doc.keys()]` which included ALL documents ever seen
  - This could incorrectly include documents from previous iterations not meant to be in the final result
- Fixed to properly handle passages accumulation:
  - When using RRF without cross-encoder: only collect ranks, don't accumulate passages
  - When using cross-encoder or neither RRF nor cross-encoder: properly accumulate passages
  - Ensures correct and consistent behavior across different configuration combinations

### 5. Fixed RRF Bias Issue
- Previously, documents not found by a retrieval method were assigned `float("inf")` as their rank
- This caused documents found by only one method to be unfairly penalized compared to documents found by multiple methods
- Now documents not found by a method get `max_rank = n_similar_chunks * retrieval_multiple`
- This ensures fair scoring while still giving some preference to documents found by multiple methods

Example of the bias that was fixed:
- Before: Document ranked #1 in semantic search only would score: 1/(1+c) ≈ 0.0164 (with c=60)
- Before: Document ranked #20 in all three methods would score: 3/(20+c) ≈ 0.0375
- The mediocre document would rank 2.3x higher despite being lower quality in each method
- After: The single-method document gets a fair chance by assigning reasonable ranks to missing methods

### 6. Updated Dependencies
- Updated all references throughout the codebase:
  - `DocChatAgent`: Uses new parameters throughout
  - `LanceDocChatAgent`: Updated to use `n_similar_chunks`
  - `ParsingConfig`: Made `n_similar_docs` Optional[int]
- Updated ruff pre-commit hook from v0.12.0 to v0.12.1

## Migration Guide

### Old Configuration
```python
config = DocChatAgentConfig(
    parsing=ParsingConfig(
        n_similar_docs=5  # This controlled both retrieval and final output
    )
)
```

### New Configuration
```python
config = DocChatAgentConfig(
    n_similar_chunks=5,    # Number of chunks each method retrieves
    n_relevant_chunks=3,   # Final number after reranking
    parsing=ParsingConfig(
        # n_similar_docs is deprecated, don't set it
    )
)
```

The new configuration provides more flexibility:
- You can retrieve more chunks initially (e.g., 10 per method)
- Then use reranking to select the best ones (e.g., top 3)
- This improves retrieval quality without increasing final context size

## Technical Details

### RRF Score Calculation (Fixed)
```python
# Old (biased) approach:
rank_semantic = id2_rank_semantic.get(id_, float("inf"))
rank_bm25 = id2_rank_bm25.get(id_, float("inf"))
rank_fuzzy = id2_rank_fuzzy.get(id_, float("inf"))

# New (fair) approach:
max_rank = self.config.n_similar_chunks * retrieval_multiple
rank_semantic = id2_rank_semantic.get(id_, max_rank)
rank_bm25 = id2_rank_bm25.get(id_, max_rank)
rank_fuzzy = id2_rank_fuzzy.get(id_, max_rank)
```

## Impact
- **Better Retrieval Quality**: The RRF fix ensures that high-quality documents found by a single method aren't unfairly discarded
- **Prevents Invalid Configurations**: Smart validation ensures users don't accidentally create setups that would produce poor results
- **Clearer Configuration**: Separating retrieval count from final output count provides more control
- **Fixes Critical Bug**: The passages accumulation fix prevents incorrect documents from appearing in results
- **Backward Compatible**: Existing code continues to work with deprecation warnings

## Related PR
- PR #874: https://github.com/langroid/langroid/pull/874