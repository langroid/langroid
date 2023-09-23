import logging
from abc import ABC, abstractmethod

from chromadb.api.types import EmbeddingFunction
from pydantic import BaseSettings

logging.getLogger("openai").setLevel(logging.ERROR)


class EmbeddingModelsConfig(BaseSettings):
    model_type: str = "openai"
    dims: int = 0


class EmbeddingModel(ABC):
    """
    Abstract base class for an embedding model.
    """

    @classmethod
    def create(cls, config: EmbeddingModelsConfig) -> "EmbeddingModel":
        from langroid.embedding_models.models import (
            OpenAIEmbeddings,
            OpenAIEmbeddingsConfig,
            SentenceTransformerEmbeddings,
            SentenceTransformerEmbeddingsConfig,
        )

        if isinstance(config, OpenAIEmbeddingsConfig):
            return OpenAIEmbeddings(config)
        elif isinstance(config, SentenceTransformerEmbeddingsConfig):
            return SentenceTransformerEmbeddings(config)
        else:
            raise ValueError(f"Unknown embedding config: {config.__repr_name__}")

    @abstractmethod
    def embedding_fn(self) -> EmbeddingFunction:
        pass

    @property
    @abstractmethod
    def embedding_dims(self) -> int:
        pass
