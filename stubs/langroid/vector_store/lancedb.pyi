from typing import Sequence

import pandas as pd
from _typeshed import Incomplete
from lancedb.query import LanceVectorQueryBuilder as LanceVectorQueryBuilder
from pydantic import BaseModel as BaseModel

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
    dataframe_to_document_model as dataframe_to_document_model,
)
from langroid.utils.pydantic_utils import (
    dataframe_to_documents as dataframe_to_documents,
)
from langroid.utils.pydantic_utils import (
    extend_document_class as extend_document_class,
)
from langroid.utils.pydantic_utils import (
    extra_metadata as extra_metadata,
)
from langroid.utils.pydantic_utils import (
    flatten_pydantic_instance as flatten_pydantic_instance,
)
from langroid.utils.pydantic_utils import (
    flatten_pydantic_model as flatten_pydantic_model,
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

has_lancedb: bool
logger: Incomplete

class LanceDBConfig(VectorStoreConfig):
    cloud: bool
    collection_name: str | None
    storage_path: str
    embedding: EmbeddingModelsConfig
    distance: str
    document_class: type[Document]
    flatten: bool

class LanceDB(VectorStore):
    config: Incomplete
    embedding_fn: Incomplete
    embedding_dim: Incomplete
    host: Incomplete
    port: Incomplete
    is_from_dataframe: bool
    df_metadata_columns: Incomplete
    client: Incomplete
    def __init__(self, config: LanceDBConfig = ...) -> None: ...
    def clear_empty_collections(self) -> int: ...
    def clear_all_collections(self, really: bool = False, prefix: str = "") -> int: ...
    def list_collections(self, empty: bool = False) -> list[str]: ...
    def create_collection(
        self, collection_name: str, replace: bool = False
    ) -> None: ...
    def add_documents(self, documents: Sequence[Document]) -> None: ...
    def add_dataframe(
        self, df: pd.DataFrame, content: str = "content", metadata: list[str] = []
    ) -> None: ...
    def delete_collection(self, collection_name: str) -> None: ...
    def get_all_documents(self, where: str = "") -> list[Document]: ...
    def get_documents_by_ids(self, ids: list[str]) -> list[Document]: ...
    def similar_texts_with_scores(
        self, text: str, k: int = 1, where: str | None = None
    ) -> list[tuple[Document, float]]: ...
