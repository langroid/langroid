# Release Notes - v0.56.9

## DocChatAgent Improvements

### Fixed Reciprocal Rank Fusion (RRF) Scoring
- Documents not found in a retrieval method now receive a rank of `max_rank + 1` instead of `max_rank`
- This ensures missing documents are properly penalized compared to documents that appear at the last position
- Improves the accuracy of RRF scoring when combining results from semantic search, BM25, and fuzzy matching

### Improved Cross-Encoder Reranking
- The `rerank_with_cross_encoder` method now only reorders passages without filtering
- Final selection of `n_relevant_chunks` is handled consistently in `get_relevant_chunks`
- This aligns cross-encoder behavior with other reranking methods (diversity, periphery)

### Simplified Conditional Logic
- Removed redundant checks for `cross_encoder_reranking_model` when `use_reciprocal_rank_fusion` is already being evaluated
- Clearer mutual exclusion between RRF and cross-encoder reranking
- Updated warning message for better clarity when both options are configured