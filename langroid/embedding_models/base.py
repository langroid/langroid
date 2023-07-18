import logging
from abc import ABC, abstractmethod

from chromadb.api.types import EmbeddingFunction
from pydantic import BaseSettings

logging.getLogger("openai").setLevel(logging.ERROR)


class EmbeddingModelsConfig(BaseSettings):
    model_type: str = "openai"


class EmbeddingModel(ABC):
    """
    Abstract base class for an embedding model.
    """

    @classmethod
    def create(cls, config: EmbeddingModelsConfig) -> "EmbeddingModel":
        from langroid.embedding_models.models import (
            OpenAIEmbeddings,
            SentenceTransformerEmbeddings,
        )

        emb_class = {
            "openai": OpenAIEmbeddings,
            "sentence-transformer": SentenceTransformerEmbeddings,
        }.get(config.model_type, OpenAIEmbeddings)
        return emb_class(config)  # type: ignore

    @abstractmethod
    def embedding_fn(self) -> EmbeddingFunction:
        pass

    @property
    @abstractmethod
    def embedding_dims(self) -> int:
        pass
