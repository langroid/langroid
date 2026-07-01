import json
import logging
import os
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

from dotenv import load_dotenv

from langroid.exceptions import LangroidImportError
from langroid.mytypes import Document
from langroid.utils.configuration import settings
from langroid.vector_store.base import VectorStore, VectorStoreConfig

logger = logging.getLogger(__name__)


has_dakera: bool = True
try:
    from dakera import DakeraClient, DistanceMetric, NotFoundError, Vector
except ImportError:
    if not TYPE_CHECKING:
        DakeraClient = Any
        DistanceMetric = Any
        Vector = Any

        class NotFoundError(Exception):  # type: ignore
            """Fallback when the ``dakera`` package is not installed."""

        has_dakera = False


# Distance metrics accepted in the config, mapped to the Dakera SDK enum.
_METRIC_ALIASES = ("cosine", "euclidean", "dot_product")


class DakeraDBConfig(VectorStoreConfig):
    collection_name: str | None = "temp"
    url: str = "http://localhost:3000"
    api_key: str = ""  # falls back to the DAKERA_API_KEY env var
    metric: str = "cosine"
    # Upper bound on the number of documents returned by `get_all_documents`,
    # which Dakera serves via a similarity query with a placeholder vector.
    max_all_documents: int = 10_000


class DakeraDB(VectorStore):
    """VectorStore backed by a self-hosted Dakera memory server.

    Dakera stores each document as a vector inside a *namespace* (the langroid
    "collection"). Embeddings are supplied by langroid's configured embedding
    model; Dakera handles storage and dense similarity search over the raw
    vector-namespace API.
    """

    def __init__(self, config: DakeraDBConfig = DakeraDBConfig()):
        super().__init__(config)
        if not has_dakera:
            raise LangroidImportError("dakera", "dakera")
        self.config: DakeraDBConfig = config
        load_dotenv()
        key = config.api_key or os.getenv("DAKERA_API_KEY", "")
        url = config.url or os.getenv("DAKERA_URL", "http://localhost:3000")
        if not key:
            raise ValueError(
                "DAKERA_API_KEY not set, could not instantiate Dakera client"
            )
        self.client = DakeraClient(base_url=url, api_key=key)

        if config.collection_name:
            self.create_collection(
                collection_name=config.collection_name,
                replace=config.replace_collection,
            )

    def _distance(self) -> "DistanceMetric":
        metric = self.config.metric
        if metric not in _METRIC_ALIASES:
            metric = "cosine"
        return DistanceMetric(metric)

    def clear_empty_collections(self) -> int:
        namespaces = self.client.list_namespaces()
        n_deletes = 0
        for ns in namespaces:
            if ns.vector_count == 0:
                self.delete_collection(collection_name=ns.name)
                n_deletes += 1
        return n_deletes

    def clear_all_collections(self, really: bool = False, prefix: str = "") -> int:
        """
        Returns:
            Number of Dakera namespaces that were deleted.

        Args:
            really: whether to really delete all matching namespaces.
            prefix: only namespaces whose name starts with this prefix are deleted.
        """
        if not really:
            logger.warning("Not deleting all collections, set really=True to confirm")
            return 0
        namespaces = [
            ns for ns in self.client.list_namespaces() if ns.name.startswith(prefix)
        ]
        if len(namespaces) == 0:
            logger.warning(f"No collections found with prefix {prefix}")
            return 0
        for ns in namespaces:
            self.delete_collection(collection_name=ns.name)
        logger.warning(f"Deleted {len(namespaces)} namespaces with prefix {prefix}")
        return len(namespaces)

    def list_collections(self, empty: bool = False) -> List[str]:
        """
        Returns:
            Names of Dakera namespaces.

        Args:
            empty: if True, include namespaces with no vectors.
        """
        namespaces = self.client.list_namespaces()
        if empty:
            return [ns.name for ns in namespaces]
        return [ns.name for ns in namespaces if ns.vector_count > 0]

    def create_collection(self, collection_name: str, replace: bool = False) -> None:
        """
        Create a namespace, optionally replacing an existing non-empty one.

        Args:
            collection_name: name of the namespace to create.
            replace: whether to replace an existing non-empty namespace.
        """
        self.config.collection_name = collection_name
        existing = {ns.name: ns for ns in self.client.list_namespaces()}
        if collection_name in existing:
            if existing[collection_name].vector_count > 0:
                logger.warning(f"Non-empty collection {collection_name} already exists")
                if not replace:
                    logger.warning("Not replacing collection")
                    return
                logger.warning("Recreating fresh collection")
            self.delete_collection(collection_name=collection_name)

        self.client.configure_namespace(
            collection_name,
            dimension=self.embedding_dim,
            distance=self._distance(),
        )

    def delete_collection(self, collection_name: str) -> None:
        logger.info(f"Attempting to delete {collection_name}")
        try:
            self.client.delete_namespace(collection_name)
        except NotFoundError:
            logger.debug(f"Namespace {collection_name} not found, nothing to delete")

    def _to_vectors(self, documents: Sequence[Document]) -> List[Any]:
        document_dicts = [doc.model_dump() for doc in documents]
        document_ids = [doc.id() for doc in documents]
        embedding_vectors = self.embedding_fn([doc.content for doc in documents])
        return [
            Vector(
                id=document_id,
                values=list(embedding_vector),
                metadata={
                    **document_dict["metadata"],
                    **{
                        key: value
                        for key, value in document_dict.items()
                        if key != "metadata"
                    },
                },
            )
            for document_dict, document_id, embedding_vector in zip(
                document_dicts, document_ids, embedding_vectors
            )
        ]

    def add_documents(self, documents: Sequence[Document]) -> None:
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot ingest docs")
        if len(documents) == 0:
            logger.warning("Empty list of documents passed into add_documents")
            return

        super().maybe_add_ids(documents)
        vectors = self._to_vectors(documents)

        if self.config.collection_name not in self.list_collections(empty=True):
            self.create_collection(
                collection_name=self.config.collection_name, replace=True
            )

        batch_size = self.config.batch_size
        for i in range(0, len(vectors), batch_size):
            self.client.upsert(
                self.config.collection_name, vectors=vectors[i : i + batch_size]
            )

    def get_all_documents(self, where: str = "") -> List[Document]:
        """
        Returns:
            All documents in the current collection, up to
            ``config.max_all_documents``.

        Args:
            where: optional Dakera metadata filter as a JSON string.
        """
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot retrieve docs")

        filter_ = json.loads(where) if where else None
        placeholder = [0.0] * self.embedding_dim
        result = self.client.query(
            self.config.collection_name,
            vector=placeholder,
            top_k=self.config.max_all_documents,
            filter=filter_,
            include_metadata=True,
        )
        docs = [self._transform(match.metadata or {}) for match in result.results]
        if len(docs) == self.config.max_all_documents:
            logger.warning(
                f"get_all_documents hit the max_all_documents limit of "
                f"{self.config.max_all_documents}; more documents may exist."
            )
        return docs

    def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
        """
        Returns:
            Documents reconstructed from the metadata stored in Dakera, in the
            same order as ``ids`` (missing ids are skipped).

        Args:
            ids: vector ids to fetch.
        """
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot retrieve docs")
        if not ids:
            return []
        vectors = self.client.fetch(
            self.config.collection_name,
            ids=ids,
            include_values=False,
            include_metadata=True,
        )
        id_mapping = {vector.id: vector for vector in vectors}
        return [
            self._transform(id_mapping[_id].metadata or {})
            for _id in ids
            if _id in id_mapping
        ]

    def similar_texts_with_scores(
        self,
        text: str,
        k: int = 1,
        where: Optional[str] = None,
        neighbors: int = 0,
    ) -> List[Tuple[Document, float]]:
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot search")
        if k < 1:
            raise ValueError(f"k for Dakera vector search must be >= 1, got {k}")

        filter_ = json.loads(where) if where else None
        query_vector = self.embedding_fn([text])[0]
        result = self.client.query(
            self.config.collection_name,
            vector=list(query_vector),
            top_k=k,
            filter=filter_,
            include_metadata=True,
            distance_metric=self._distance(),
        )
        doc_score_pairs = [
            (self._transform(match.metadata or {}), match.score)
            for match in result.results
        ]
        if settings.debug and doc_score_pairs:
            max_score = max(pair[1] for pair in doc_score_pairs)
            logger.info(f"Found {len(doc_score_pairs)} matches, max score: {max_score}")
        self.show_if_debug(doc_score_pairs)
        return doc_score_pairs

    def _transform(self, metadata_dict: Dict[str, Any]) -> Document:
        """Reconstruct a Document from the metadata stored alongside its vector."""
        return self.config.document_class(
            **{**metadata_dict, "metadata": {**metadata_dict}}
        )
