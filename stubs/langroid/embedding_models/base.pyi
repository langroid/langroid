import abc
from abc import ABC, abstractmethod

from pydantic import BaseSettings

from langroid.mytypes import EmbeddingFunction as EmbeddingFunction

class EmbeddingModelsConfig(BaseSettings):
    model_type: str
    dims: int
    context_length: int
    batch_size: int

class EmbeddingModel(ABC, metaclass=abc.ABCMeta):
    @classmethod
    def create(cls, config: EmbeddingModelsConfig) -> EmbeddingModel: ...
    @abstractmethod
    def embedding_fn(self) -> EmbeddingFunction: ...
    @property
    @abstractmethod
    def embedding_dims(self) -> int: ...
    def similarity(self, text1: str, text2: str) -> float: ...
