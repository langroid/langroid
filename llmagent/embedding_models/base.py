from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from chromadb.api.types import EmbeddingFunction

logging.getLogger("openai").setLevel(logging.ERROR)


@dataclass
class EmbeddingModelsConfig:
    model_type: str = "openai"


class EmbeddingModel(ABC):
    # factory method
    @classmethod
    def create(cls, config: EmbeddingModelsConfig):
        from llmagent.embedding_models.models import (
            OpenAIEmbeddings,
            SentenceTransformerEmbeddings,
        )

        emb_class = {
            "openai": OpenAIEmbeddings,
            "sentence-transformer": SentenceTransformerEmbeddings,
        }.get(config.model_type, OpenAIEmbeddings)
        return emb_class(config)

    @abstractmethod
    def embedding_fn(self) -> EmbeddingFunction:
        pass

    @abstractmethod
    def embedding_dims(self) -> int:
        pass
