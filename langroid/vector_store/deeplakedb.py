import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

from langroid.embedding_models.base import EmbeddingModelsConfig
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.exceptions import LangroidImportError
from langroid.mytypes import Document
from langroid.utils.configuration import settings
from langroid.vector_store.base import VectorStore, VectorStoreConfig

try:
    import deeplake

    has_deeplake = True
except ImportError:
    has_deeplake = False

logger = logging.getLogger(__name__)


class DeepLakeDBConfig(VectorStoreConfig):
    collection_name: str | None = "temp"
    storage_path: str = ".deeplake/data"
    distance_metric: str = "cos"
    embedding: EmbeddingModelsConfig = OpenAIEmbeddingsConfig()


class DeepLakeDB(VectorStore):
    """Vector store backed by DeepLake (https://github.com/activeloopai/deeplake)."""

    def __init__(self, config: DeepLakeDBConfig = DeepLakeDBConfig()):
        super().__init__(config)
        if not has_deeplake:
            raise LangroidImportError("deeplake", "deeplake")
        self.config: DeepLakeDBConfig = config
        load_dotenv()
        self.token = os.getenv("ACTIVELOOP_TOKEN")
        self.client: Optional["deeplake.VectorStore"] = None
        if self.config.collection_name is not None:
            self.create_collection(
                self.config.collection_name,
                replace=self.config.replace_collection,
            )

    def _collection_path(self, collection_name: str) -> str:
        """Path of the given collection, relative to `storage_path`."""
        base = self.config.storage_path.rstrip("/")
        return f"{base}/{collection_name}"

    def _collection_names(self) -> List[str]:
        """Names of collections found under `storage_path`."""
        base = self.config.storage_path
        if "://" in base or not os.path.isdir(base):
            # e.g. hub://... paths - no way to list datasets under these
            return []
        return [
            name
            for name in os.listdir(base)
            if deeplake.exists(os.path.join(base, name), token=self.token)
        ]

    def clear_empty_collections(self) -> int:
        n_deletes = 0
        for name in self._collection_names():
            ds = deeplake.load(
                self._collection_path(name), verbose=False, token=self.token
            )
            if len(ds) == 0:
                n_deletes += 1
                self.delete_collection(name)
        return n_deletes

    def clear_all_collections(self, really: bool = False, prefix: str = "") -> int:
        """Clear all collections with the given prefix."""
        if not really:
            logger.warning("Not deleting all collections, set really=True to confirm")
            return 0
        names = [n for n in self._collection_names() if n.startswith(prefix)]
        if len(names) == 0:
            logger.warning(f"No collections found with prefix {prefix}")
            return 0
        n_empty_deletes, n_non_empty_deletes = 0, 0
        for name in names:
            ds = deeplake.load(
                self._collection_path(name), verbose=False, token=self.token
            )
            is_empty = len(ds) == 0
            self.delete_collection(name)
            n_empty_deletes += is_empty
            n_non_empty_deletes += not is_empty
        logger.warning(
            f"""
            Deleted {n_empty_deletes} empty collections and
            {n_non_empty_deletes} non-empty collections.
            """
        )
        return n_empty_deletes + n_non_empty_deletes

    def list_collections(self, empty: bool = False) -> List[str]:
        """
        Returns:
            List of collection names that have at least one vector
            (all names, if empty=True).
        """
        names = self._collection_names()
        if empty:
            return names
        result = []
        for name in names:
            ds = deeplake.load(
                self._collection_path(name), verbose=False, token=self.token
            )
            if len(ds) > 0:
                result.append(name)
        return result

    def create_collection(self, collection_name: str, replace: bool = False) -> None:
        self.config.replace_collection = replace
        self.config.collection_name = collection_name
        path = self._collection_path(collection_name)
        self.client = deeplake.VectorStore(
            path=path,
            overwrite=replace,
            token=self.token,
            verbose=False,
        )

    def add_documents(self, documents: Sequence[Document]) -> None:
        super().maybe_add_ids(documents)
        if len(documents) == 0:
            return
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot ingest docs")
        if self.client is None:
            self.create_collection(self.config.collection_name, replace=False)
        assert self.client is not None

        contents = [doc.content for doc in documents]
        metadata_dicts = [doc.metadata.dict_bool_int() for doc in documents]
        ids = [str(d.id()) for d in documents]
        embedding_vecs = self.embedding_fn(contents)

        b = self.config.batch_size
        for i in range(0, len(documents), b):
            self.client.add(
                text=contents[i : i + b],
                metadata=metadata_dicts[i : i + b],
                embedding=embedding_vecs[i : i + b],
                id=ids[i : i + b],
            )

    def delete_collection(self, collection_name: str) -> None:
        path = self._collection_path(collection_name)
        try:
            if deeplake.exists(path, token=self.token):
                deeplake.delete(path, token=self.token, force=True)
        except Exception as e:
            logger.warning(f"Error deleting DeepLake collection {collection_name}: {e}")
        if self.config.collection_name == collection_name:
            self.client = None

    def _docs_from_result(self, result: Dict[str, Any]) -> List[Document]:
        if len(result.get("id", [])) == 0:
            return []
        return [
            self.config.document_class(
                content=text,
                metadata=self.config.metadata_class(**metadata),
            )
            for text, metadata in zip(result["text"], result["metadata"])
        ]

    def get_all_documents(self, where: str = "") -> List[Document]:
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot retrieve docs")
        if self.client is None or len(self.client) == 0:
            return []
        filter_dict = json.loads(where) if where else None
        result = self.client.search(
            filter={"metadata": filter_dict} if filter_dict else (lambda x: True),
            k=len(self.client),
        )
        return self._docs_from_result(result)

    def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot retrieve docs")
        if self.client is None or len(self.client) == 0 or len(ids) == 0:
            return []
        _ids = set(str(id) for id in ids)
        result = self.client.search(
            filter=lambda x: x["id"].data()["value"] in _ids,
            k=len(self.client),
        )
        docs_by_id = dict(zip(result.get("id", []), self._docs_from_result(result)))
        return [docs_by_id[id] for id in ids if id in docs_by_id]

    def similar_texts_with_scores(
        self,
        text: str,
        k: int = 1,
        where: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        if self.client is None or len(self.client) == 0:
            logger.warning(f"No documents in collection, cannot search for {text}")
            return []
        embedding = self.embedding_fn([text])[0]
        filter_dict = json.loads(where) if where else None
        result = self.client.search(
            embedding=embedding,
            k=min(k, len(self.client)),
            filter={"metadata": filter_dict} if filter_dict else None,
            distance_metric=self.config.distance_metric,
        )
        docs = self._docs_from_result(result)
        if len(docs) == 0:
            logger.warning(f"No matches found for {text}")
            return []
        scores = result.get("score", [1.0] * len(docs))
        if settings.debug:
            logger.info(f"Found {len(docs)} matches, max score: {max(scores)}")
        doc_score_pairs = list(zip(docs, scores))
        self.show_if_debug(doc_score_pairs)
        return doc_score_pairs
