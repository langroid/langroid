import json
import logging
import os
import re
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from dotenv import load_dotenv

from langroid import LangroidImportError
from langroid.mytypes import Document

# import dataclass
from langroid.pydantic_v1 import BaseModel
from langroid.utils.configuration import settings
from langroid.vector_store.base import VectorStore, VectorStoreConfig

logger = logging.getLogger(__name__)


has_pinecone: bool = True
try:
    from pinecone import Pinecone, PineconeApiException, ServerlessSpec
except ImportError:

    if not TYPE_CHECKING:

        class ServerlessSpec(BaseModel):
            """
            Fallback Serverless specification configuration to avoid import errors.
            """

            cloud: str
            region: str

        PineconeApiException = Any  # type: ignore
        Pinecone = Any  # type: ignore
        has_pinecone = False


@dataclass(frozen=True)
class IndexMeta:
    name: str
    total_vector_count: int


class PineconeDBConfig(VectorStoreConfig):
    cloud: bool = True
    collection_name: str | None = "temp"
    spec: ServerlessSpec = ServerlessSpec(cloud="aws", region="us-east-1")
    deletion_protection: Literal["enabled", "disabled"] | None = None
    metric: str = "cosine"
    pagination_size: int = 100


class PineconeDB(VectorStore):
    def __init__(self, config: PineconeDBConfig = PineconeDBConfig()):
        super().__init__(config)
        if not has_pinecone:
            raise LangroidImportError("pinecone", "pinecone")
        self.config: PineconeDBConfig = config
        load_dotenv()
        key = os.getenv("PINECONE_API_KEY")

        if not key:
            raise ValueError("PINECONE_API_KEY not set, could not instantiate client")
        self.client = Pinecone(api_key=key)

        if config.collection_name:
            self.create_collection(
                collection_name=config.collection_name,
                replace=config.replace_collection,
            )

    def clear_empty_collections(self) -> int:
        indexes = self._list_index_metas(empty=True)
        n_deletes = 0
        for index in indexes:
            if index.total_vector_count == -1:
                logger.warning(
                    f"Error fetching details for {index.name} when scanning indexes"
                )
            n_deletes += 1
            self.delete_collection(collection_name=index.name)
        return n_deletes

    def clear_all_collections(self, really: bool = False, prefix: str = "") -> int:
        """
        Returns:
            Number of Pinecone indexes that were deleted

        Args:
            really: Optional[bool] - whether to really delete all Pinecone collections
            prefix: Optional[str] - string to match potential Pinecone
                indexes for deletion
        """
        if not really:
            logger.warning("Not deleting all collections, set really=True to confirm")
            return 0
        indexes = [
            c for c in self._list_index_metas(empty=True) if c.name.startswith(prefix)
        ]
        if len(indexes) == 0:
            logger.warning(f"No collections found with prefix {prefix}")
            return 0
        n_empty_deletes, n_non_empty_deletes = 0, 0
        for index_desc in indexes:
            self.delete_collection(collection_name=index_desc.name)
            n_empty_deletes += index_desc.total_vector_count == 0
            n_non_empty_deletes += index_desc.total_vector_count > 0
        logger.warning(
            f"""
            Deleted {n_empty_deletes} empty indexes and
            {n_non_empty_deletes} non-empty indexes
            """
        )
        return n_empty_deletes + n_non_empty_deletes

    def list_collections(self, empty: bool = False) -> List[str]:
        """
        Returns:
            List of Pinecone indices that have at least one vector.

        Args:
            empty: Optional[bool] - whether to include empty collections
        """
        indexes = self.client.list_indexes()
        res: List[str] = []
        if empty:
            res.extend(indexes.names())
            return res

        for index in indexes.names():
            index_meta = self.client.Index(name=index)
            if index_meta.describe_index_stats().get("total_vector_count", 0) > 0:
                res.append(index)
        return res

    def _list_index_metas(self, empty: bool = False) -> List[IndexMeta]:
        """
        Returns:
            List of objects describing Pinecone indices

        Args:
            empty: Optional[bool] - whether to include empty collections
        """
        indexes = self.client.list_indexes()
        res = []
        for index in indexes.names():
            index_meta = self._fetch_index_meta(index)
            if empty:
                res.append(index_meta)
            elif index_meta.total_vector_count > 0:
                res.append(index_meta)
        return res

    def _fetch_index_meta(self, index_name: str) -> IndexMeta:
        """
        Returns:
            A dataclass describing the input Index by name and vector count
            to save a bit on index description calls

        Args:
            index_name: str - Name of the index in Pinecone
        """
        try:
            index = self.client.Index(name=index_name)
            stats = index.describe_index_stats()
            return IndexMeta(
                name=index_name, total_vector_count=stats.get("total_vector_count", 0)
            )
        except PineconeApiException as e:
            logger.warning(f"Error fetching details for index {index_name}")
            logger.warning(e)
            return IndexMeta(name=index_name, total_vector_count=-1)

    def create_collection(self, collection_name: str, replace: bool = False) -> None:
        """
        Create a collection with the given name, optionally replacing an existing
        collection if `replace` is True.

        Args:
            collection_name: str - Configuration of the collection to create.
            replace: Optional[Bool] - Whether to replace an existing collection
                with the same name. Defaults to False.
        """
        pattern = re.compile(r"^[a-z0-9-]+$")
        if not pattern.match(collection_name):
            raise ValueError(
                "Pinecone index names must be lowercase alphanumeric characters or '-'"
            )
        self.config.collection_name = collection_name
        if collection_name in self.list_collections(empty=True):
            index = self.client.Index(name=collection_name)
            stats = index.describe_index_stats()
            status = self.client.describe_index(name=collection_name)
            if status["status"]["ready"] and stats["total_vector_count"] > 0:
                logger.warning(f"Non-empty collection {collection_name} already exists")
                if not replace:
                    logger.warning("Not replacing collection")
                    return
                else:
                    logger.warning("Recreating fresh collection")
            self.delete_collection(collection_name=collection_name)

        payload = {
            "name": collection_name,
            "dimension": self.embedding_dim,
            "spec": self.config.spec,
            "metric": self.config.metric,
            "timeout": self.config.timeout,
        }

        if self.config.deletion_protection:
            payload["deletion_protection"] = self.config.deletion_protection

        try:
            self.client.create_index(**payload)
        except PineconeApiException as e:
            logger.error(e)

    def delete_collection(self, collection_name: str) -> None:
        logger.info(f"Attempting to delete {collection_name}")
        try:
            self.client.delete_index(name=collection_name)
        except PineconeApiException as e:
            logger.error(f"Failed to delete {collection_name}")
            logger.error(e)

    def add_documents(self, documents: Sequence[Document], namespace: str = "") -> None:
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot ingest docs")

        if len(documents) == 0:
            logger.warning("Empty list of documents passed into add_documents")
            return

        super().maybe_add_ids(documents)
        document_dicts = [doc.dict() for doc in documents]
        document_ids = [doc.id() for doc in documents]
        embedding_vectors = self.embedding_fn([doc.content for doc in documents])
        vectors = [
            {
                "id": document_id,
                "values": embedding_vector,
                "metadata": {
                    **document_dict["metadata"],
                    **{
                        key: value
                        for key, value in document_dict.items()
                        if key != "metadata"
                    },
                },
            }
            for document_dict, document_id, embedding_vector in zip(
                document_dicts, document_ids, embedding_vectors
            )
        ]

        if self.config.collection_name not in self.list_collections(empty=True):
            self.create_collection(
                collection_name=self.config.collection_name, replace=True
            )

        index = self.client.Index(name=self.config.collection_name)
        batch_size = self.config.batch_size

        for i in range(0, len(documents), batch_size):
            try:
                if namespace:
                    index.upsert(
                        vectors=vectors[i : i + batch_size], namespace=namespace
                    )
                else:
                    index.upsert(vectors=vectors[i : i + batch_size])
            except PineconeApiException as e:
                logger.error(
                    f"Unable to add of docs between indices {i} and {batch_size}"
                )
                logger.error(e)

    def get_all_documents(
        self, prefix: str = "", namespace: str = ""
    ) -> List[Document]:
        """
        Returns:
            All documents for the collection currently defined in
            the configuration object

        Args:
            prefix: str - document id prefix to search for
            namespace: str - partition of vectors to search within the index
        """
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot retrieve docs")
        docs = []

        request_filters: Dict[str, Union[str, int]] = {
            "limit": self.config.pagination_size
        }
        if prefix:
            request_filters["prefix"] = prefix
        if namespace:
            request_filters["namespace"] = namespace

        index = self.client.Index(name=self.config.collection_name)

        while True:
            response = index.list_paginated(**request_filters)
            vectors = response.get("vectors", [])

            if not vectors:
                logger.warning("Received empty list while requesting for vector ids")
                logger.warning("Halting fetch requests")
                if settings.debug:
                    logger.debug(f"Request for failed fetch was: {request_filters}")
                break

            docs.extend(
                self.get_documents_by_ids(
                    ids=[vector.get("id") for vector in vectors],
                    namespace=namespace if namespace else "",
                )
            )

            pagination_token = response.get("pagination", {}).get("next", None)

            if not pagination_token:
                break

            request_filters["pagination_token"] = pagination_token

        return docs

    def get_documents_by_ids(
        self, ids: List[str], namespace: str = ""
    ) -> List[Document]:
        """
        Returns:
            Fetches document text embedded in Pinecone index metadata

        Args:
            ids: List[str] - vector data object ids to retrieve
            namespace: str - partition of vectors to search within the index
        """
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot retrieve docs")
        index = self.client.Index(name=self.config.collection_name)

        if namespace:
            records = index.fetch(ids=ids, namespace=namespace)
        else:
            records = index.fetch(ids=ids)

        id_mapping = {key: value for key, value in records["vectors"].items()}
        ordered_payloads = [id_mapping[_id] for _id in ids if _id in id_mapping]
        return [
            self.transform_pinecone_vector(payload.get("metadata", {}))
            for payload in ordered_payloads
        ]

    def similar_texts_with_scores(
        self,
        text: str,
        k: int = 1,
        where: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot search")

        if k < 1 or k > 9999:
            raise ValueError(
                f"TopK for Pinecone vector search must be 1 < k < 10000, k was {k}"
            )

        vector_search_request = {
            "top_k": k,
            "include_metadata": True,
            "vector": self.embedding_fn([text])[0],
        }
        if where:
            vector_search_request["filter"] = json.loads(where) if where else None
        if namespace:
            vector_search_request["namespace"] = namespace

        index = self.client.Index(name=self.config.collection_name)
        response = index.query(**vector_search_request)
        doc_score_pairs = [
            (
                self.transform_pinecone_vector(match.get("metadata", {})),
                match.get("score", 0),
            )
            for match in response.get("matches", [])
        ]
        if settings.debug:
            max_score = max([pair[1] for pair in doc_score_pairs])
            logger.info(f"Found {len(doc_score_pairs)} matches, max score: {max_score}")
        self.show_if_debug(doc_score_pairs)
        return doc_score_pairs

    def transform_pinecone_vector(self, metadata_dict: Dict[str, Any]) -> Document:
        """
        Parses the metadata response from the Pinecone vector query and
        formats it into a dictionary that can be parsed by the Document class
        associated with the PineconeDBConfig class

        Returns:
            Well formed dictionary object to be transformed into a Document

        Args:
            metadata_dict: Dict - the metadata dictionary from the Pinecone
                vector query match
        """
        return self.config.document_class(
            **{**metadata_dict, "metadata": {**metadata_dict}}
        )
