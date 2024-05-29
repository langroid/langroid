import abc
from abc import ABC, abstractmethod
from typing import Sequence

from _typeshed import Incomplete
from pydantic import BaseSettings

from langroid.embedding_models.base import (
    EmbeddingModel as EmbeddingModel,
)
from langroid.embedding_models.base import (
    EmbeddingModelsConfig as EmbeddingModelsConfig,
)
from langroid.embedding_models.models import (
    OpenAIEmbeddingsConfig as OpenAIEmbeddingsConfig,
)
from langroid.mytypes import Document as Document
from langroid.utils.algorithms.graph import (
    components as components,
)
from langroid.utils.algorithms.graph import (
    topological_sort as topological_sort,
)
from langroid.utils.configuration import settings as settings
from langroid.utils.output.printing import print_long_text as print_long_text
from langroid.utils.pandas_utils import stringify as stringify

logger: Incomplete

class VectorStoreConfig(BaseSettings):
    type: str
    collection_name: str | None
    replace_collection: bool
    storage_path: str
    cloud: bool
    batch_size: int
    embedding: EmbeddingModelsConfig
    timeout: int
    host: str
    port: int

class VectorStore(ABC, metaclass=abc.ABCMeta):
    config: Incomplete
    embedding_model: Incomplete
    def __init__(self, config: VectorStoreConfig) -> None: ...
    @staticmethod
    def create(config: VectorStoreConfig) -> VectorStore | None: ...
    @abstractmethod
    def clear_empty_collections(self) -> int: ...
    @abstractmethod
    def clear_all_collections(self, really: bool = False, prefix: str = "") -> int: ...
    @abstractmethod
    def list_collections(self, empty: bool = False) -> list[str]: ...
    def set_collection(self, collection_name: str, replace: bool = False) -> None: ...
    @abstractmethod
    def create_collection(
        self, collection_name: str, replace: bool = False
    ) -> None: ...
    @abstractmethod
    def add_documents(self, documents: Sequence[Document]) -> None: ...
    def compute_from_docs(self, docs: list[Document], calc: str) -> str: ...
    def maybe_add_ids(self, documents: Sequence[Document]) -> None: ...
    @abstractmethod
    def similar_texts_with_scores(
        self, text: str, k: int = 1, where: str | None = None
    ) -> list[tuple[Document, float]]: ...
    def add_context_window(
        self, docs_scores: list[tuple[Document, float]], neighbors: int = 0
    ) -> list[tuple[Document, float]]: ...
    @staticmethod
    def remove_overlaps(windows: list[list[str]]) -> list[list[str]]: ...
    @abstractmethod
    def get_all_documents(self, where: str = "") -> list[Document]: ...
    @abstractmethod
    def get_documents_by_ids(self, ids: list[str]) -> list[Document]: ...
    @abstractmethod
    def delete_collection(self, collection_name: str) -> None: ...
    def show_if_debug(self, doc_score_pairs: list[tuple[Document, float]]) -> None: ...
