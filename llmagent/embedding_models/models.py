import os
from typing import Callable, List

import openai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from llmagent.embedding_models.base import EmbeddingModel, EmbeddingModelsConfig
from llmagent.language_models.utils import retry_with_exponential_backoff
from llmagent.mytypes import Embeddings


class OpenAIEmbeddingsConfig(EmbeddingModelsConfig):
    model_name: str = "text-embedding-ada-002"
    api_key: str = ""
    dims: int = 1536


class SentenceTransformerEmbeddingsConfig(EmbeddingModelsConfig):
    model_name: str = "all-MiniLM-L6-v2"
    dims: int = 384


class OpenAIEmbeddings(EmbeddingModel):
    def __init__(self, config: OpenAIEmbeddingsConfig):
        super().__init__()
        self.config = config
        load_dotenv()
        self.config.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.config.api_key

    def embedding_fn(self) -> Callable[[List[str]], Embeddings]:
        @retry_with_exponential_backoff
        def fn(texts: List[str]) -> Embeddings:
            result = openai.Embedding.create(input=texts, model=self.config.model_name)
            return [d["embedding"] for d in result["data"]]

        return fn

    @property
    def embedding_dims(self) -> int:
        return self.config.dims


class SentenceTransformerEmbeddings(EmbeddingModel):
    def __init__(self, config: SentenceTransformerEmbeddingsConfig):
        super().__init__()
        self.config = config
        self.model = SentenceTransformer(self.config.model_name)

    def embedding_fn(self) -> Callable[[List[str]], Embeddings]:
        def fn(texts: List[str]) -> Embeddings:
            return self.model.encode(texts, convert_to_numpy=True).tolist()

        return fn

    @property
    def embedding_dims(self) -> int:
        return self.config.dims


def embedding_model(embedding_fn_type: str = "openai") -> EmbeddingModel:
    """
    Thin wrapper around chromadb.utils.embedding_functions.
    Args:
        embedding_fn_type: "openai" or "sentencetransformer" # others soon
    Returns:
        EmbeddingModel
    """
    if embedding_fn_type == "openai":
        return OpenAIEmbeddings
    else:  # default sentence transformer
        return SentenceTransformerEmbeddings
