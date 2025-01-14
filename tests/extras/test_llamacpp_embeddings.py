"""
Test for HuggingFace embeddings.
This depends on sentence-transformers being installed:
 uv sync --dev --extra hf-embeddings
"""

from langroid.embedding_models.base import EmbeddingModel
from langroid.embedding_models.models import LlamaCppServerEmbeddingsConfig

"""
    Pytest for llama.cpp server acting as the embeddings host.

    You can find an example of how to run llama.cpp server as an embeddings host in
    docs/notes/llama-cpp-embeddings.md
    
    You must fill out the following variables or the test will fail:

    embedding_address       - This is a string containing the IP address and 
                              port of the llama.cpp server 
                              e.g. "http://localhost:51060"
    embed_context_length    - This is the context length of the model you have
                              loaded into llama.cpp server
    embedding_dimensions    - The dimensions of the embeddings returned from
                              the model.

"""

embedding_address: str = "http://localhost:51060"
embed_context_length: int = 2048
embedding_dimensions: int = 768


def test_embeddings():
    sentence_cfg = LlamaCppServerEmbeddingsConfig(
        api_base=embedding_address,
        context_length=embed_context_length,
        batch_size=embed_context_length,
        dims=embedding_dimensions,
    )

    sentence_model = EmbeddingModel.create(sentence_cfg)

    sentence_fn = sentence_model.embedding_fn()

    assert len(sentence_fn(["hello"])[0]) == sentence_model.embedding_dims
