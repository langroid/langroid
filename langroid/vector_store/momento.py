"""
Momento Vector Index.
https://docs.momentohq.com/vector-index/develop/api-reference
DEPRECATED: API is unstable.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence, Tuple, no_type_check

from dotenv import load_dotenv

from langroid.exceptions import LangroidImportError

try:
    import momento.responses.vector_index as mvi_response
    from momento import (
        # PreviewVectorIndexClientAsync,
        CredentialProvider,
        PreviewVectorIndexClient,
        VectorIndexConfigurations,
    )
    from momento.requests.vector_index import (
        ALL_METADATA,
        Item,
        SimilarityMetric,
    )

    has_momento = True
except ImportError:
    has_momento = False


from langroid.embedding_models.base import (
    EmbeddingModel,
    EmbeddingModelsConfig,
)
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.mytypes import Document, EmbeddingFunction
from langroid.utils.configuration import settings
from langroid.utils.pydantic_utils import (
    flatten_pydantic_instance,
    nested_dict_from_flat,
)
from langroid.vector_store.base import VectorStore, VectorStoreConfig

logger = logging.getLogger(__name__)


class MomentoVIConfig(VectorStoreConfig):
    cloud: bool = True
    collection_name: str | None = "temp"
    embedding: EmbeddingModelsConfig = OpenAIEmbeddingsConfig()


class MomentoVI(VectorStore):
    def __init__(self, config: MomentoVIConfig = MomentoVIConfig()):
        super().__init__(config)
        if not has_momento:
            raise LangroidImportError("momento", "momento")
        self.distance = SimilarityMetric.COSINE_SIMILARITY
        self.config: MomentoVIConfig = config
        emb_model = EmbeddingModel.create(config.embedding)
        self.embedding_fn: EmbeddingFunction = emb_model.embedding_fn()
        self.embedding_dim = emb_model.embedding_dims
        self.host = config.host
        self.port = config.port
        load_dotenv()
        api_key = os.getenv("MOMENTO_API_KEY")
        if config.cloud:
            if api_key is None:
                raise ValueError(
                    """MOMENTO_API_KEY env variable must be set to 
                    MomentoVI hosted service. Please set this in your .env file. 
                    """
                )
            self.client = PreviewVectorIndexClient(
                configuration=VectorIndexConfigurations.Default.latest(),
                credential_provider=CredentialProvider.from_string(api_key),
            )
        else:
            raise NotImplementedError("MomentoVI local not available yet")

        # Note: Only create collection if a non-null collection name is provided.
        # This is useful to delay creation of vecdb until we have a suitable
        # collection name (e.g. we could get it from the url or folder path).
        if config.collection_name is not None:
            self.create_collection(
                config.collection_name, replace=config.replace_collection
            )

    def clear_empty_collections(self) -> int:
        logger.warning(
            """
            Momento VI does not yet have a way to easily get size of indices,
            so clear_empty_collections is not deleting any indices.
            """
        )
        return 0

    def clear_all_collections(self, really: bool = False, prefix: str = "") -> int:
        """Clear all collections with the given prefix."""

        if not really:
            logger.warning("Not deleting all collections, set really=True to confirm")
            return 0
        coll_names = self.list_collections(empty=False)
        coll_names = [name for name in coll_names if name.startswith(prefix)]
        if len(coll_names) == 0:
            logger.warning(f"No collections found with prefix {prefix}")
            return 0
        for name in coll_names:
            self.delete_collection(name)
        logger.warning(
            f"""
            Deleted {len(coll_names)} indices from Momento VI
            """
        )
        return len(coll_names)

    def list_collections(self, empty: bool = False) -> List[str]:
        """
        Returns:
            List of collection names that have at least one vector.

        Args:
            empty (bool, optional): Whether to include empty collections.
        """
        if not has_momento:
            raise LangroidImportError("momento", "momento")
        response = self.client.list_indexes()
        if isinstance(response, mvi_response.ListIndexes.Success):
            return [ind.name for ind in response.indexes]
        elif isinstance(response, mvi_response.ListIndexes.Error):
            raise ValueError(f"Error listing collections: {response.message}")
        else:
            raise ValueError(f"Unexpected response: {response}")

    def create_collection(self, collection_name: str, replace: bool = False) -> None:
        """
        Create a collection with the given name, optionally replacing an existing
            collection if `replace` is True.
        Args:
            collection_name (str): Name of the collection to create.
            replace (bool): Whether to replace an existing collection
                with the same name. Defaults to False.
        """
        if not has_momento:
            raise LangroidImportError("momento", "momento")
        self.config.collection_name = collection_name
        response = self.client.create_index(
            index_name=collection_name,
            num_dimensions=self.embedding_dim,
            similarity_metric=self.distance,
        )
        if isinstance(response, mvi_response.CreateIndex.Success):
            logger.info(f"Created collection {collection_name}")
        elif isinstance(response, mvi_response.CreateIndex.IndexAlreadyExists):
            logger.warning(f"Collection {collection_name} already exists")
        elif isinstance(response, mvi_response.CreateIndex.Error):
            raise ValueError(
                f"Error creating collection {collection_name}: {response.message}"
            )
        if settings.debug:
            level = logger.getEffectiveLevel()
            logger.setLevel(logging.INFO)
            logger.info(f"Collection {collection_name} created")
            logger.setLevel(level)

    def add_documents(self, documents: Sequence[Document]) -> None:
        super().maybe_add_ids(documents)
        if len(documents) == 0:
            return
        embedding_vecs = self.embedding_fn([doc.content for doc in documents])
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot ingest docs")

        self.create_collection(self.config.collection_name, replace=True)

        items = [
            Item(
                id=str(d.id()),
                vector=embedding_vecs[i],
                metadata=flatten_pydantic_instance(d, force_str=True),
                # force all values to str since Momento requires it
            )
            for i, d in enumerate(documents)
        ]

        # don't insert all at once, batch in chunks of b,
        # else we get an API error
        b = self.config.batch_size
        for i in range(0, len(documents), b):
            response = self.client.upsert_item_batch(
                index_name=self.config.collection_name,
                items=items[i : i + b],
            )
            if isinstance(response, mvi_response.UpsertItemBatch.Success):
                continue
            elif isinstance(response, mvi_response.UpsertItemBatch.Error):
                raise ValueError(f"Error adding documents: {response.message}")
            else:
                raise ValueError(f"Unexpected response: {response}")

    def delete_collection(self, collection_name: str) -> None:
        delete_response = self.client.delete_index(collection_name)
        if isinstance(delete_response, mvi_response.DeleteIndex.Success):
            logger.warning(f"Deleted index {collection_name}")
        elif isinstance(delete_response, mvi_response.DeleteIndex.Error):
            logger.error(
                f"Error while deleting index {collection_name}: "
                f" {delete_response.message}"
            )

    def _to_int_or_uuid(self, id: str) -> int | str:
        try:
            return int(id)
        except ValueError:
            return id

    def get_all_documents(self, where: str = "") -> List[Document]:
        raise NotImplementedError(
            """
            MomentoVI does not support get_all_documents().
            Please use a different vector database, e.g. qdrant or chromadb.
            """
        )

    def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
        raise NotImplementedError(
            """
            MomentoVI does not support get_documents_by_ids.
            Please use a different vector database, e.g. qdrant or chromadb.
            """
        )

    @no_type_check
    def similar_texts_with_scores(
        self,
        text: str,
        k: int = 1,
        where: Optional[str] = None,
        neighbors: int = 0,  # ignored
    ) -> List[Tuple[Document, float]]:
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot search")
        embedding = self.embedding_fn([text])[0]
        response = self.client.search(
            index_name=self.config.collection_name,
            query_vector=embedding,
            top_k=k,
            metadata_fields=ALL_METADATA,
        )

        if isinstance(response, mvi_response.Search.Error):
            logger.warning(
                f"Error while searching on index {self.config.collection_name}:"
                f" {response.message}"
            )
            return []
        elif not isinstance(response, mvi_response.Search.Success):
            logger.warning(f"Unexpected response: {response}")
            return []

        scores = [match.metadata["distance"] for match in response.hits]
        docs = [
            Document.parse_obj(nested_dict_from_flat(match.metadata))
            for match in response.hits
            if match is not None
        ]
        if len(docs) == 0:
            logger.warning(f"No matches found for {text}")
            return []
        if settings.debug:
            logger.info(f"Found {len(docs)} matches, max score: {max(scores)}")
        doc_score_pairs = list(zip(docs, scores))
        self.show_if_debug(doc_score_pairs)
        return doc_score_pairs
