from abc import  ABCMeta
import logging
from chromadb.api.types import EmbeddingFunction
logging.getLogger("openai").setLevel(logging.ERROR)


class EmbeddingModel(metaclass=ABCMeta):
    @classmethod
    def embedding_fn(cls) -> EmbeddingFunction:
        pass

    @classmethod
    @property
    def embedding_dims(cls) -> int:
        pass

