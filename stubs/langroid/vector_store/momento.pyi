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
from langroid.mytypes import (
    Document as Document,
)
from langroid.mytypes import (
    EmbeddingFunction as EmbeddingFunction,
)
from langroid.utils.configuration import settings as settings
from langroid.utils.pydantic_utils import (
    flatten_pydantic_instance as flatten_pydantic_instance,
)
from langroid.utils.pydantic_utils import (
    nested_dict_from_flat as nested_dict_from_flat,
)
from langroid.vector_store.base import (
    VectorStore as VectorStore,
)
from langroid.vector_store.base import (
    VectorStoreConfig as VectorStoreConfig,
)

has_momento: bool
logger: Incomplete

class MomentoVIConfig(VectorStoreConfig):
    cloud: bool
    collection_name: str | None
    embedding: EmbeddingModelsConfig

class MomentoVI(VectorStore):
    distance: Incomplete
    config: Incomplete
    embedding_fn: Incomplete
    embedding_dim: Incomplete
    host: Incomplete
    port: Incomplete
    client: Incomplete
    def __init__(self, config: MomentoVIConfig = ...) -> None: ...
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
        self, text, k: int = 1, where: Incomplete | None = None, neighbors: int = 0
    ): ...
