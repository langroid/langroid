import logging
import os
import re
from typing import Any, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

from langroid.embedding_models.base import (
    EmbeddingModelsConfig,
)
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.exceptions import LangroidImportError
from langroid.mytypes import DocMetaData, Document
from langroid.utils.configuration import settings
from langroid.vector_store.base import VectorStore, VectorStoreConfig

logger = logging.getLogger(__name__)


class VectorDistances:
    """
    Fallback class when weaviate is not installed, to avoid import errors.
    """

    COSINE: str = "cosine"
    DOTPRODUCT: str = "dot"
    L2: str = "l2"


class WeaviateDBConfig(VectorStoreConfig):
    collection_name: str | None = "temp"
    embedding: EmbeddingModelsConfig = OpenAIEmbeddingsConfig()
    distance: str = VectorDistances.COSINE
    cloud: bool = False
    docker: bool = False
    host: str = "127.0.0.1"
    port: int = 8080
    storage_path: str = ".weaviate_embedded/data"


class WeaviateDB(VectorStore):
    def __init__(self, config: WeaviateDBConfig = WeaviateDBConfig()):
        super().__init__(config)
        try:
            import weaviate
            from weaviate.classes.init import Auth
        except ImportError:
            raise LangroidImportError("weaviate", "weaviate")

        self.config: WeaviateDBConfig = config
        load_dotenv()
        if self.config.docker:
            self.client = weaviate.connect_to_local(
                host=self.config.host,
                port=self.config.port,
            )
            self.config.cloud = False
        elif self.config.cloud:
            key = os.getenv("WEAVIATE_API_KEY")
            url = os.getenv("WEAVIATE_API_URL")
            if url is None or key is None:
                raise ValueError(
                    """WEAVIATE_API_KEY, WEAVIATE_API_URL env variables must be set to 
                    use WeaviateDB in cloud mode. Please set these values
                    in your .env file.
                    """
                )
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=url,
                auth_credentials=Auth.api_key(key),
            )
        else:
            self.client = weaviate.connect_to_embedded(
                version="latest", persistence_data_path=self.config.storage_path
            )

        if config.collection_name is not None:
            WeaviateDB.validate_and_format_collection_name(config.collection_name)

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
        non_empty_colls = [
            coll_name
            for coll_name in colls.keys()
            if len(self.client.collections.get(coll_name)) > 0
        ]

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
        try:
            from weaviate.classes.config import (
                Configure,
                VectorDistances,
            )
        except ImportError:
            raise LangroidImportError("weaviate", "weaviate")
        collection_name = WeaviateDB.validate_and_format_collection_name(
            collection_name
        )
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
        if isinstance(self.config.embedding, OpenAIEmbeddingsConfig):
            vectorizer_config = Configure.Vectorizer.text2vec_openai(
                model=self.config.embedding.model_name,
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
            for i, doc_dict in enumerate(document_dicts):
                id = doc_dict["metadata"].pop("id", None)
                batch.add_object(properties=doc_dict, uuid=id, vector=embedding_vecs[i])

    def get_all_documents(self, where: str = "") -> List[Document]:
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot retrieve docs")
        # cannot use filter as client does not support json type queries
        coll = self.client.collections.get(self.config.collection_name)
        return [self.weaviate_obj_to_doc(item) for item in coll.iterator()]

    def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
        from weaviate.classes.query import Filter

        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot retrieve docs")

        docs = []
        coll_name = self.client.collections.get(self.config.collection_name)

        result = coll_name.query.fetch_objects(
            filters=Filter.by_property("_id").contains_any(ids), limit=len(coll_name)
        )

        id_to_doc = {}
        for item in result.objects:
            doc = self.weaviate_obj_to_doc(item)
            id_to_doc[doc.metadata.id] = doc

        # Reconstruct the list of documents in the original order of input ids
        docs = [id_to_doc[id] for id in ids if id in id_to_doc]

        return docs

    def similar_texts_with_scores(
        self, text: str, k: int = 1, where: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        from weaviate.classes.query import MetadataQuery

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
        maybe_distances = [item.metadata.distance for item in response.objects]
        similarities = [0 if d is None else 1 - d for d in maybe_distances]
        docs = [self.weaviate_obj_to_doc(item) for item in response.objects]
        return list(zip(docs, similarities))

    def _create_valid_uuid_id(self, id: str) -> Any:
        from weaviate.util import generate_uuid5, get_valid_uuid

        try:
            id = get_valid_uuid(id)
            return id
        except Exception:
            return generate_uuid5(id)

    def weaviate_obj_to_doc(self, input_object: Any) -> Document:
        from weaviate.util import get_valid_uuid

        content = input_object.properties.get("content", "")
        metadata_dict = input_object.properties.get("metadata", {})

        window_ids = metadata_dict.pop("window_ids", [])
        window_ids = [str(uuid) for uuid in window_ids]

        # Ensure the id is a valid UUID string
        id_value = get_valid_uuid(input_object.uuid)

        metadata = DocMetaData(id=id_value, window_ids=window_ids, **metadata_dict)

        return Document(content=content, metadata=metadata)

    @staticmethod
    def validate_and_format_collection_name(name: str) -> str:
        """
        Formats the collection name to comply with Weaviate's naming rules:
        - Name must start with a capital letter.
        - Name can only contain letters, numbers, and underscores.
        - Replaces invalid characters with underscores.
        """
        if not name:
            raise ValueError("Collection name cannot be empty.")

        formatted_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)

        # Ensure the first letter is capitalized
        if not formatted_name[0].isupper():
            formatted_name = formatted_name.capitalize()

        # Check if the name now meets the criteria
        if not re.match(r"^[A-Z][A-Za-z0-9_]*$", formatted_name):
            raise ValueError(
                f"Invalid collection name '{name}'."
                " Names must start with a capital letter "
                "and contain only letters, numbers, and underscores."
            )

        if formatted_name != name:
            logger.warning(
                f"Collection name '{name}' was reformatted to '{formatted_name}' "
                "to comply with Weaviate's rules."
            )

        return formatted_name

    def __del__(self) -> None:
        # Gracefully close the connection with local client
        if not self.config.cloud:
            self.client.close()
