# Release Notes for v0.56.7

## DocChatAgent Improvements

- Fixed test failures caused by deprecated `n_similar_docs` parameter interfering with `n_similar_chunks` and `n_relevant_chunks` settings
- Set `n_similar_docs` default to `None` to prevent backward compatibility code from overriding intended retrieval configurations
- Optimized reciprocal rank fusion passage selection using list slicing for better performance

## Bug Fixes

- Resolved issue where `n_similar_docs=4` (old default) was silently overriding test configurations that expected 3 chunks