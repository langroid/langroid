import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple, TypeVar

import weaviate
from dotenv import load_dotenv
from weaviate.classes.config import (
    Configure,
    VectorDistances,
)
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.util import generate_uuid5, get_valid_uuid

from langroid.embedding_models.base import (
    EmbeddingModelsConfig,
)
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.mytypes import DocMetaData, Document, EmbeddingFunction, Embeddings
from langroid.utils.configuration import settings
from langroid.vector_store.base import VectorStore, VectorStoreConfig

logger = logging.getLogger(__name__)


class WeaviateDBConfig(VectorStoreConfig):
    cloud: bool = True
    collection_name: str | None = "temp"
    storage_path: str = ".weaviate/data"
    embedding: EmbeddingModelsConfig = OpenAIEmbeddingsConfig()
    distance: str = VectorDistances.COSINE


class WeaviateDB(VectorStore):
    def __init__(self, config: WeaviateDBConfig = WeaviateDBConfig()):
        super().__init__(config)
        self.config: WeaviateDBConfig = config
        self.embedding_fn: EmbeddingFunction = self.embedding_model.embedding_fn()
        self.embedding_dim = self.embedding_model.embedding_dims
        load_dotenv()
        key = os.getenv("WEAVIATE_API_KEY")
        url = os.getenv("WEAVIATE_API_URL")
        if config.cloud and None in [key, url]:
            logger.warning(
                """WEAVIATE_API_KEY, WEAVIATE_API_URL env variable must be set to use
                WeaviateDB in cloud mode. Please set these values
                in your .env file.
                """
            )
            config.cloud = False
        if config.cloud:
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=url,
                auth_credentials=Auth.api_key(key),
            )
        if config.collection_name is not None:
             if config.collection_name[0].islower():
                logger.warning(
                    f"""Beware that WeaviateDB collection names always start with first
                            letter capitalized so creating collection name with
                            {config.collection_name[0].upper()
                             + config.collection_name[1:]}
                    """
                    )

    def clear_empty_collections(self) -> int:
        colls = self.client.collections.list_all()
        n_deletes = 0
        for coll_name, _ in colls.items():
            val = self.client.collections.get(coll_name)
            if len(val) == 0:
                n_deletes += 1
                self.client.collections.delete(coll_name)
        return n_deletes

    def list_collections(self, empty: bool = False) -> List[str]:
        colls = self.client.collections.list_all()
        if empty:
            return list(colls.keys())
        non_empty_colls = []
        for coll_name in colls.keys():
            if len(self.client.collections.get(coll_name)) > 0:
                non_empty_colls.append(coll_name)
        return non_empty_colls

    def clear_all_collections(self, really: bool = False, prefix: str = "") -> int:
        if not really:
            logger.warning(
                "Not really deleting all collections ,set really=True to confirm"
            )
            return 0
        coll_names = [
            c for c in self.list_collections(empty=True) if c.startswith(prefix)
        ]
        if len(coll_names) == 0:
            logger.warning(f"No collections found with prefix {prefix}")
            return 0
        n_empty_deletes = 0
        n_non_empty_deletes = 0
        for name in coll_names:
            info = self.client.collections.get(name)
            points_count = len(info)

            n_empty_deletes += points_count == 0
            n_non_empty_deletes += points_count > 0
            self.client.collections.delete(name)
        logger.warning(
            f"""
            Deleted {n_empty_deletes} empty collections and
            {n_non_empty_deletes} non-empty collections.
            """
        )
        return n_empty_deletes + n_non_empty_deletes

    def delete_collection(self, collection_name: str) -> None:
        self.client.collections.delete(name=collection_name)

    def create_collection(self, collection_name: str, replace: bool = False) -> None:
         # Capitalize the first letter if necessary
        if collection_name and collection_name[0].islower():
            collection_name = collection_name[0].upper() + collection_name[1:]
        self.config.collection_name = collection_name
        if self.client.collections.exists(name=collection_name):
            coll = self.client.collections.get(name=collection_name)
            if len(coll) > 0:
                logger.warning(f"Non-empty Collection {collection_name} already exists")
                if not replace:
                    logger.warning("Not replacing collection")
                    return
                else:
                    logger.warning("Recreating fresh collection")
            self.client.collections.delete(name=collection_name)

        vector_index_config = Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE,
        )
        if self.config.embedding == OpenAIEmbeddingsConfig:
            vectorizer_config = Configure.Vectorizer.text2vec_openai(
                model=self.embedding_model
            )
        else:
            vectorizer_config = None

        collection_info = self.client.collections.create(
            name=collection_name,
            vector_index_config=vector_index_config,
            vectorizer_config=vectorizer_config,
        )
        collection_info = self.client.collections.get(name=collection_name)
        assert len(collection_info) in [0, None]
        if settings.debug:
            level = logger.getEffectiveLevel()
            logger.setLevel(logging.INFO)
            logger.info(collection_info)
            logger.setLevel(level)

    def add_documents(self, documents: Sequence[Document]) -> None:
        super().maybe_add_ids(documents)
        colls = self.list_collections(empty=True)
        for doc in documents:
            doc.metadata.id = str(self._create_valid_uuid_id(doc.metadata.id))
        if len(documents) == 0:
            return

        document_dicts = [doc.dict() for doc in documents]
        embedding_vecs = self.embedding_fn([doc.content for doc in documents])
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot ingest docs")
        if self.config.collection_name not in colls:
            self.create_collection(self.config.collection_name, replace=True)
        coll_name = self.client.collections.get(self.config.collection_name)
        with coll_name.batch.dynamic() as batch:
            for i, doc in enumerate(document_dicts):
                id = doc["metadata"].pop("id", None)
                batch.add_object(properties=doc, uuid=id, vector=embedding_vecs[i])

    def get_all_documents(self, where: str = "") -> List[Document]:
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot retrieve docs")
        docs = []
        # cannot use filter as client does not support json type queries
        coll = self.client.collections.get(self.config.collection_name)
        for item in coll.iterator():
            docs.append(self.weaviate_obj_to_doc(item))
        return docs

    def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot retrieve docs")
        
        docs = []
        coll_name = self.client.collections.get(self.config.collection_name)
        
        result = coll_name.query.fetch_objects(
            filters=Filter.by_property("_id").contains_any(ids),
            limit=len(coll_name)
        )
        
        # Create a dictionary mapping IDs to documents
        id_to_doc = {}
        for item in result.objects:
            doc = self.weaviate_obj_to_doc(item)
            id_to_doc[doc.metadata.id] = doc
        
        # Reconstruct the list of documents in the original order of input ids
        for id in ids:
            if id in id_to_doc:
                docs.append(id_to_doc[id])
        
        return docs

    def similar_texts_with_scores(
        self, text: str, k: int = 1, where: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        embedding = self.embedding_fn([text])[0]
        if self.config.collection_name is None:
            raise ValueError("No collections name set,cannot search")
        coll = self.client.collections.get(self.config.collection_name)
        response = coll.query.near_vector(
            near_vector=embedding,
            limit=k,
            return_properties=True,
            return_metadata=MetadataQuery(distance=True),
        )
        docs = []
        distances = []
        for item in response.objects:
            docs.append(self.weaviate_obj_to_doc(item))
            distances.append(1 - item.metadata.distance)
        doc_score_pairs = list(zip(docs, distances))
        return doc_score_pairs

    def _create_valid_uuid_id(self, id: str) -> str:
        try:
            id = get_valid_uuid(id)
            return id
        except Exception:
            return generate_uuid5(id)

    def weaviate_obj_to_doc(self, input_object) -> Document:
        content = input_object.properties.get("content", "")
        metadata_dict = input_object.properties.get("metadata", {})

        source = metadata_dict.get("source", "")
        is_chunk = metadata_dict.get("is_chunk", False)
        window_ids = metadata_dict.get("window_ids", [])
        window_ids = [str(uuid) for uuid in window_ids]


        # Ensure the id is a valid UUID string
        id_value = get_valid_uuid(input_object.uuid)

        # Construct the MyDocMetaData object
        metadata = DocMetaData(
            source=source,
            is_chunk=is_chunk,
            id=id_value,
            window_ids=window_ids,
        )

        # Construct and return the MyDoc object
        return Document(content=content, metadata=metadata)
