from typing import Sequence

from _typeshed import Incomplete
from meilisearch_python_sdk.index import AsyncIndex as AsyncIndex
from meilisearch_python_sdk.models.documents import DocumentsInfo as DocumentsInfo

from langroid.exceptions import LangroidImportError as LangroidImportError
from langroid.mytypes import DocMetaData as DocMetaData
from langroid.mytypes import Document as Document
from langroid.utils.configuration import settings as settings
from langroid.vector_store.base import (
    VectorStore as VectorStore,
)
from langroid.vector_store.base import (
    VectorStoreConfig as VectorStoreConfig,
)

logger: Incomplete

class MeiliSearchConfig(VectorStoreConfig):
    cloud: bool
    collection_name: str | None
    primary_key: str
    port: int

class MeiliSearch(VectorStore):
    config: Incomplete
    host: Incomplete
    port: Incomplete
    key: Incomplete
    url: Incomplete
    client: Incomplete
    def __init__(self, config: MeiliSearchConfig = ...) -> None: ...
    def clear_empty_collections(self) -> int: ...
    def clear_all_collections(self, really: bool = False, prefix: str = "") -> int: ...
    def list_collections(self, empty: bool = False) -> list[str]: ...
    def create_collection(
        self, collection_name: str, replace: bool = False
    ) -> None: ...
    def add_documents(self, documents: Sequence[Document]) -> None: ...
    def delete_collection(self, collection_name: str) -> None: ...
    def get_all_documents(self, where: str = "") -> list[Document]: ...
    def get_documents_by_ids(self, ids: list[str]) -> list[Document]: ...
    def similar_texts_with_scores(
        self, text: str, k: int = 20, where: str | None = None, neighbors: int = 0
    ) -> list[tuple[Document, float]]: ...
