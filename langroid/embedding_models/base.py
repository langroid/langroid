import logging
from abc import ABC, abstractmethod

import numpy as np

from langroid.mytypes import EmbeddingFunction
from langroid.pydantic_v1 import BaseSettings

logging.getLogger("openai").setLevel(logging.ERROR)


class EmbeddingModelsConfig(BaseSettings):
    model_type: str = "openai"
    dims: int = 0
    context_length: int = 512
    batch_size: int = 512


class EmbeddingModel(ABC):
    """
    Abstract base class for an embedding model.
    """

    @classmethod
    def create(cls, config: EmbeddingModelsConfig) -> "EmbeddingModel":
        from langroid.embedding_models.models import (
            AzureOpenAIEmbeddings,
            AzureOpenAIEmbeddingsConfig,
            FastEmbedEmbeddings,
            FastEmbedEmbeddingsConfig,
            GeminiEmbeddings,
            GeminiEmbeddingsConfig,
            LlamaCppServerEmbeddings,
            LlamaCppServerEmbeddingsConfig,
            OpenAIEmbeddings,
            OpenAIEmbeddingsConfig,
            SentenceTransformerEmbeddings,
            SentenceTransformerEmbeddingsConfig,
        )
        from langroid.embedding_models.remote_embeds import (
            RemoteEmbeddings,
            RemoteEmbeddingsConfig,
        )

        if isinstance(config, RemoteEmbeddingsConfig):
            return RemoteEmbeddings(config)
        elif isinstance(config, OpenAIEmbeddingsConfig):
            return OpenAIEmbeddings(config)
        elif isinstance(config, AzureOpenAIEmbeddingsConfig):
            return AzureOpenAIEmbeddings(config)
        elif isinstance(config, SentenceTransformerEmbeddingsConfig):
            return SentenceTransformerEmbeddings(config)
        elif isinstance(config, FastEmbedEmbeddingsConfig):
            return FastEmbedEmbeddings(config)
        elif isinstance(config, LlamaCppServerEmbeddingsConfig):
            return LlamaCppServerEmbeddings(config)
        elif isinstance(config, GeminiEmbeddingsConfig):
            return GeminiEmbeddings(config)
        else:
            raise ValueError(f"Unknown embedding config: {config.__repr_name__}")

    @abstractmethod
    def embedding_fn(self) -> EmbeddingFunction:
        pass

    @property
    @abstractmethod
    def embedding_dims(self) -> int:
        pass

    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        [emb1, emb2] = self.embedding_fn()([text1, text2])
        return float(
            np.array(emb1)
            @ np.array(emb2)
            / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        )
