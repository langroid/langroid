"""
Momento Vector Index.
https://docs.momentohq.com/vector-index/develop/api-reference
"""
import logging
import os
from typing import List, Optional, Sequence, Tuple

from chromadb.api.types import EmbeddingFunction
from dotenv import load_dotenv
from momento import (
    # PreviewVectorIndexClientAsync,
    CredentialProvider,
    PreviewVectorIndexClient,
    VectorIndexConfigurations,
)
from momento.requests.vector_index import SimilarityMetric
from momento.responses.vector_index import CreateIndex

from langroid.embedding_models.base import (
    EmbeddingModel,
    EmbeddingModelsConfig,
)
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.mytypes import Document
from langroid.utils.configuration import settings
from langroid.vector_store.base import VectorStore, VectorStoreConfig

logger = logging.getLogger(__name__)


class MomentoVIConfig(VectorStoreConfig):
    cloud: bool = True
    collection_name: str | None = None
    embedding: EmbeddingModelsConfig = OpenAIEmbeddingsConfig()
    distance: str = SimilarityMetric.COSINE_SIMILARITY


class MomentoVI(VectorStore):
    def __init__(self, config: MomentoVIConfig):
        super().__init__(config)
        self.config: MomentoVIConfig = config
        emb_model = EmbeddingModel.create(config.embedding)
        self.embedding_fn: EmbeddingFunction = emb_model.embedding_fn()
        self.embedding_dim = emb_model.embedding_dims
        self.host = config.host
        self.port = config.port
        load_dotenv()
        key = os.getenv("MOMENTO_API_KEY")
        if config.cloud and key is None:
            raise ValueError(
                """MOMENTO_API_KEY env variable must be set to 
                MomentoVI hosted service. Please set this in your .env file. 
                """
            )
        if config.cloud:
            self.client = PreviewVectorIndexClient(
                configuration=VectorIndexConfigurations.Default.latest(),
                credential_provider=CredentialProvider.from_environment_variable(
                    "MOMENTO_API_KEY"
                ),
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
            self.client.delete_index(index_name=name)
        logger.warning(
            f"""
            Deleted {len(coll_names)} indices from Momento VIk
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
        colls = list(self.client.list_indexes())
        if empty:
            return [coll.name for coll in colls]
        counts = [
            self.client.get_collection(collection_name=coll.name).points_count
            for coll in colls
        ]
        return [coll.name for coll, count in zip(colls, counts) if count > 0]

    def create_collection(self, collection_name: str, replace: bool = False) -> None:
        """
        Create a collection with the given name, optionally replacing an existing
            collection if `replace` is True.
        Args:
            collection_name (str): Name of the collection to create.
            replace (bool): Whether to replace an existing collection
                with the same name. Defaults to False.
        """
        self.config.collection_name = collection_name
        response = self.client.create_index(
            index_name=collection_name,
            num_dimensions=self.embedding_dim,
            similarity_metric=self.config.distance,
        )
        match response:
            case CreateIndex.Success():
                logger.info(f"Created collection {collection_name}")
            case CreateIndex.IndexAlreadyExists():
                logger.warning(f"Collection {collection_name} already exists")
            case CreateIndex.Error(error):
                raise ValueError(
                    f"Error creating collection {collection_name}: {error}"
                )
        if settings.debug:
            level = logger.getEffectiveLevel()
            logger.setLevel(logging.INFO)
            logger.info(f"Collection {collection_name} created")
            logger.setLevel(level)

    def add_documents(self, documents: Sequence[Document]) -> None:
        from momento.requests.vector_index import Item
        from momento.responses.vector_index import UpsertItemBatch

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
                metadata=d,
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
            match response:
                case UpsertItemBatch.Success():
                    pass
                case UpsertItemBatch.Error(_):
                    raise response.inner_exception
                case _:
                    raise ValueError("Unexpected response")

    def delete_collection(self, collection_name: str) -> None:
        self.client.delete_collection(collection_name=collection_name)

    def _to_int_or_uuid(self, id: str) -> int | str:
        try:
            return int(id)
        except ValueError:
            return id

    def get_all_documents(self) -> List[Document]:
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot retrieve docs")
        docs = []
        offset = 0
        while True:
            results, next_page_offset = self.client.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=None,
                offset=offset,
                limit=10_000,  # try getting all at once, if not we keep paging
                with_payload=True,
                with_vectors=False,
            )
            docs += [Document(**record.payload) for record in results]  # type: ignore
            if next_page_offset is None:
                break
            offset = next_page_offset  # type: ignore
        return docs

    def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot retrieve docs")
        _ids = [self._to_int_or_uuid(id) for id in ids]
        records = self.client.retrieve(
            collection_name=self.config.collection_name,
            ids=_ids,
            with_vectors=False,
            with_payload=True,
        )
        docs = [Document(**record.payload) for record in records]  # type: ignore
        return docs

    def similar_texts_with_scores(
        self,
        text: str,
        k: int = 1,
        where: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        return []
