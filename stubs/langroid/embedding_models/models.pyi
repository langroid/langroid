from typing import Callable

from _typeshed import Incomplete

from langroid.embedding_models.base import (
    EmbeddingModel as EmbeddingModel,
)
from langroid.embedding_models.base import (
    EmbeddingModelsConfig as EmbeddingModelsConfig,
)
from langroid.mytypes import Embeddings as Embeddings
from langroid.parsing.utils import batched as batched

class OpenAIEmbeddingsConfig(EmbeddingModelsConfig):
    model_type: str
    model_name: str
    api_key: str
    api_base: str | None
    organization: str
    dims: int
    context_length: int

class SentenceTransformerEmbeddingsConfig(EmbeddingModelsConfig):
    model_type: str
    model_name: str
    context_length: int
    data_parallel: bool
    device: str | None
    devices: list[str] | None

class EmbeddingFunctionCallable:
    model: Incomplete
    batch_size: Incomplete
    def __init__(self, model: OpenAIEmbeddings, batch_size: int = 512) -> None: ...
    def __call__(self, input: list[str]) -> Embeddings: ...

class OpenAIEmbeddings(EmbeddingModel):
    config: Incomplete
    client: Incomplete
    tokenizer: Incomplete
    def __init__(self, config: OpenAIEmbeddingsConfig = ...) -> None: ...
    def truncate_texts(self, texts: list[str]) -> list[list[int]]: ...
    def embedding_fn(self) -> Callable[[list[str]], Embeddings]: ...
    @property
    def embedding_dims(self) -> int: ...

STEC = SentenceTransformerEmbeddingsConfig

class SentenceTransformerEmbeddings(EmbeddingModel):
    config: Incomplete
    model: Incomplete
    pool: Incomplete
    tokenizer: Incomplete
    def __init__(self, config: STEC = ...) -> None: ...
    def embedding_fn(self) -> Callable[[list[str]], Embeddings]: ...
    @property
    def embedding_dims(self) -> int: ...

def embedding_model(embedding_fn_type: str = "openai") -> EmbeddingModel: ...
