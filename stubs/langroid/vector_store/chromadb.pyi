from typing import Sequence

from _typeshed import Incomplete

from langroid.embedding_models.base import (
    EmbeddingModel as EmbeddingModel,
)
from langroid.embedding_models.base import (
    EmbeddingModelsConfig as EmbeddingModelsConfig,
)
from langroid.embedding_models.models import (
    OpenAIEmbeddingsConfig as OpenAIEmbeddingsConfig,
)
from langroid.exceptions import LangroidImportError as LangroidImportError
from langroid.mytypes import DocMetaData as DocMetaData
from langroid.mytypes import Document as Document
from langroid.utils.configuration import settings as settings
from langroid.utils.output.printing import print_long_text as print_long_text
from langroid.vector_store.base import (
    VectorStore as VectorStore,
)
from langroid.vector_store.base import (
    VectorStoreConfig as VectorStoreConfig,
)

logger: Incomplete

class ChromaDBConfig(VectorStoreConfig):
    collection_name: str
    storage_path: str
    embedding: EmbeddingModelsConfig
    host: str
    port: int

class ChromaDB(VectorStore):
    config: Incomplete
    embedding_fn: Incomplete
    client: Incomplete
    def __init__(self, config: ChromaDBConfig = ...) -> None: ...
    def clear_all_collections(self, really: bool = False, prefix: str = "") -> int: ...
    def clear_empty_collections(self) -> int: ...
    def list_collections(self, empty: bool = False) -> list[str]: ...
    collection: Incomplete
    def create_collection(
        self, collection_name: str, replace: bool = False
    ) -> None: ...
    def add_documents(self, documents: Sequence[Document]) -> None: ...
    def get_all_documents(self, where: str = "") -> list[Document]: ...
    def get_documents_by_ids(self, ids: list[str]) -> list[Document]: ...
    def delete_collection(self, collection_name: str) -> None: ...
    def similar_texts_with_scores(
        self, text: str, k: int = 1, where: str | None = None
    ) -> list[tuple[Document, float]]: ...
