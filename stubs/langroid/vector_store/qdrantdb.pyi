from typing import Sequence, TypeVar

from _typeshed import Incomplete
from qdrant_client.conversions.common_types import ScoredPoint as ScoredPoint
from qdrant_client.http.models import SparseVector

from langroid.embedding_models.base import (
    EmbeddingModel as EmbeddingModel,
)
from langroid.embedding_models.base import (
    EmbeddingModelsConfig as EmbeddingModelsConfig,
)
from langroid.embedding_models.models import (
    OpenAIEmbeddingsConfig as OpenAIEmbeddingsConfig,
)
from langroid.mytypes import (
    Document as Document,
)
from langroid.mytypes import (
    EmbeddingFunction as EmbeddingFunction,
)
from langroid.mytypes import (
    Embeddings as Embeddings,
)
from langroid.utils.configuration import settings as settings
from langroid.vector_store.base import (
    VectorStore as VectorStore,
)
from langroid.vector_store.base import (
    VectorStoreConfig as VectorStoreConfig,
)

logger: Incomplete
T = TypeVar("T")

def from_optional(x: T | None, default: T) -> T: ...
def is_valid_uuid(uuid_to_test: str) -> bool: ...

class QdrantDBConfig(VectorStoreConfig):
    cloud: bool
    collection_name: str | None
    storage_path: str
    embedding: EmbeddingModelsConfig
    distance: str
    use_sparse_embeddings: bool
    sparse_embedding_model: str
    sparse_limit: int

class QdrantDB(VectorStore):
    config: Incomplete
    embedding_fn: Incomplete
    embedding_dim: Incomplete
    sparse_tokenizer: Incomplete
    sparse_model: Incomplete
    host: Incomplete
    port: Incomplete
    client: Incomplete
    def __init__(self, config: QdrantDBConfig = ...) -> None: ...
    def clear_empty_collections(self) -> int: ...
    def clear_all_collections(self, really: bool = False, prefix: str = "") -> int: ...
    def list_collections(self, empty: bool = False) -> list[str]: ...
    def create_collection(
        self, collection_name: str, replace: bool = False
    ) -> None: ...
    def get_sparse_embeddings(self, inputs: list[str]) -> list[SparseVector]: ...
    def add_documents(self, documents: Sequence[Document]) -> None: ...
    def delete_collection(self, collection_name: str) -> None: ...
    def get_all_documents(self, where: str = "") -> list[Document]: ...
    def get_documents_by_ids(self, ids: list[str]) -> list[Document]: ...
    def similar_texts_with_scores(
        self, text: str, k: int = 1, where: str | None = None, neighbors: int = 0
    ) -> list[tuple[Document, float]]: ...
