# Issue #919: llama.cpp Embeddings Support

## Background

User reported issues using llama.cpp server for local embeddings with Langroid. The error occurred when using `LlamaCppServerEmbeddingsConfig`:

```
TypeError: list indices must be integers or slices, not str
```

This happened at line 466 in `langroid/embedding_models/models.py`:
```python
embeddings = response.json()["embedding"]
```

## Investigation Summary

### Can llama.cpp Generate Embeddings?

**YES!** llama.cpp supports embeddings in two ways:

1. **Dedicated embedding models** (RECOMMENDED):
   - nomic-embed-text-v1.5 (768 dims)
   - nomic-embed-text-v2-moe
   - nomic-embed-code
   - Other GGUF embedding models

2. **Regular LLMs** (works but not optimal):
   - gpt-oss-20b, gpt-oss-120b
   - Llama models
   - By extracting internal representations

### How to Enable

Start llama-server with the `--embeddings` flag:

```bash
./llama-server -ngl 100 -c 2048 \
  -m ~/nomic-embed-text-v1.5.Q8_0.gguf \
  --host localhost --port 8080 \
  --embeddings -b 2048 -ub 2048
```

## llama.cpp Embedding Endpoints

llama.cpp provides multiple embedding endpoints with different response formats:

### 1. Native `/embedding` endpoint

**Request:**
```json
{
  "content": "your text here"
}
```

**Response:**
```json
{
  "embedding": [0.1, 0.2, 0.3, ...]
}
```

### 2. OpenAI-compatible `/v1/embeddings` endpoint

**Request:**
```json
{
  "input": "your text here",
  "model": "model-name"
}
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, 0.3, ...],
      "index": 0
    }
  ],
  "model": "model-name",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

## The Problem

The original Langroid code expected only format #1 (native):
```python
embeddings = response.json()["embedding"]
```

However, llama.cpp can return **different formats** depending on:
- Endpoint used (`/embedding` vs `/v1/embeddings`)
- Server version/configuration
- Batch mode settings

The error indicated that `response.json()` returned a **list**, not a **dict**, suggesting llama.cpp returned an array format.

## Discovered Response Formats

Through investigation, we identified **5 possible response formats**:

1. **Native format**: `{"embedding": [floats]}`
2. **Array format**: `[{"embedding": [floats]}]`
3. **Double-nested**: `[{"embedding": [[floats]]}]`
4. **OpenAI-compatible**: `{"data": [{"embedding": [floats]}]}`
5. **Dict-nested**: `{"embedding": [[floats]]}`

## Our Solution

### Implementation

Added a robust `_extract_embedding()` method in `langroid/embedding_models/models.py` (lines 483-544) that:

1. Tries each format in order
2. Validates the extracted embedding is a list of floats
3. Provides clear error messages if format is unrecognized

```python
def _extract_embedding(
    self, response_json: dict[str, Any] | list[Any]
) -> List[int | float]:
    """
    Extract embedding vector from llama.cpp response.

    Handles multiple response formats:
    1. Native /embedding: {"embedding": [floats]}
    2. Array format: [{"embedding": [floats]}]
    3. Double-nested: [{"embedding": [[floats]]}]
    4. OpenAI /v1/embeddings: {"data": [{"embedding": [floats]}]}
    5. Nested in dict: {"embedding": [[floats]]}

    Args:
        response_json: The JSON response from llama.cpp server

    Returns:
        List of floats representing the embedding vector

    Raises:
        ValueError: If response format is not recognized
    """
    # Implementation handles all 5 formats...
```

### Modified `generate_embedding()` method

Changed from:
```python
embeddings = response.json()["embedding"]
```

To:
```python
embeddings = self._extract_embedding(response.json())
```

## Testing

Created comprehensive unit tests in `tests/extras/test_llamacpp_embedding_formats.py`:

- ✅ test_native_format
- ✅ test_array_format
- ✅ test_double_nested_array_format
- ✅ test_openai_compatible_format
- ✅ test_nested_in_dict_format
- ✅ test_invalid_format_raises_error
- ✅ test_generate_embedding_with_native_format (mocked)
- ✅ test_generate_embedding_with_array_format (mocked)
- ✅ test_generate_embedding_with_openai_format (mocked)
- ✅ test_generate_embedding_http_error

**All tests pass** ✅
**Linting and type checking pass** ✅

## Comparison with PR #920

### PR #920 Approach

Changed:
```python
embeddings = response.json()["embedding"]
```

To:
```python
embeddings = response.json()[0]["embedding"][0]
```

### Issues with PR #920

1. **Too specific**: Only handles ONE format: `[{"embedding": [[floats]]}]`
2. **Logic error**: The double `[0]` indexing would extract a single float, not the full embedding vector
3. **Would fail validation**: The existing validation expects a list of floats
4. **No tests**: No unit tests provided
5. **No documentation**: No explanation of what format is expected

### Our Solution Advantages

1. **Handles 5 different formats** automatically
2. **Backwards compatible**: Works with existing deployments
3. **Well-tested**: 10 unit tests covering all scenarios
4. **Well-documented**: Clear docstring explaining all formats
5. **Robust error messages**: Helps users debug configuration issues

## Usage Example

### Configuration

```python
from langroid.embedding_models.models import LlamaCppServerEmbeddingsConfig
from langroid.vector_store.qdrantdb import QdrantDBConfig

embed_cfg = LlamaCppServerEmbeddingsConfig(
    api_base="http://localhost:8080",  # Your llama.cpp server
    dims=768,  # Match your embedding model dimensions
    context_length=2048,
    batch_size=2048,
)

vecdb_config = QdrantDBConfig(
    collection_name="my-docs",
    embedding=embed_cfg,
    storage_path=".qdrant/",
)
```

### Running llama-server

```bash
# For dedicated embedding model (RECOMMENDED)
./llama-server -ngl 100 -c 2048 \
  -m ~/nomic-embed-text-v1.5.Q8_0.gguf \
  --embeddings -b 2048 -ub 2048 \
  --host localhost --port 8080

# For LLM-based embeddings (gpt-oss example)
./llama-server -ngl 99 \
  -m ~/.cache/llama.cpp/gpt-oss-20b.gguf \
  --embeddings \
  --host localhost --port 8080
```

## Recommendations

### For Users

1. **Use dedicated embedding models** like nomic-embed-text-v1.5 for best results
2. **Match dimensions** in config to your embedding model
3. **Use the `--embeddings` flag** when starting llama-server
4. **Check server logs** if you encounter issues

### For Langroid

1. ✅ **Implemented**: Robust format detection in `_extract_embedding()`
2. ✅ **Tested**: Comprehensive unit tests
3. ✅ **Documented**: Clear docstrings and examples
4. **Consider**: Adding example in `examples/docqa/` using local embeddings
5. **Consider**: Adding to documentation/tutorials

## Files Modified

- `langroid/embedding_models/models.py` - Added `_extract_embedding()` method
- `tests/extras/test_llamacpp_embedding_formats.py` - New comprehensive test suite

## References

- Issue #919: https://github.com/langroid/langroid/issues/919
- PR #920: https://github.com/langroid/langroid/pull/920
- llama.cpp discussion #7712: https://github.com/ggml-org/llama.cpp/discussions/7712
- nomic-embed models: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF
- Langroid docs: `docs/notes/llama-cpp-embeddings.md`

## Conclusion

**Issue #919 is now resolved** with a robust, well-tested solution that handles all known llama.cpp embedding response formats. Users can now use local embeddings with llama.cpp without worrying about response format variations.

**PR #920 is not needed** as our solution is more comprehensive and handles all cases, not just one specific format.
