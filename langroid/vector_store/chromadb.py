import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langroid.embedding_models.base import (
    EmbeddingModel,
    EmbeddingModelsConfig,
)
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.exceptions import LangroidImportError
from langroid.mytypes import Document
from langroid.utils.configuration import settings
from langroid.utils.output.printing import print_long_text
from langroid.vector_store.base import VectorStore, VectorStoreConfig

logger = logging.getLogger(__name__)


class ChromaDBConfig(VectorStoreConfig):
    collection_name: str = "temp"
    storage_path: str = ".chroma/data"
    embedding: EmbeddingModelsConfig = OpenAIEmbeddingsConfig()
    host: str = "127.0.0.1"
    port: int = 6333


class ChromaDB(VectorStore):
    def __init__(self, config: ChromaDBConfig = ChromaDBConfig()):
        super().__init__(config)
        try:
            import chromadb
        except ImportError:
            raise LangroidImportError("chromadb", "chromadb")
        self.config = config
        emb_model = EmbeddingModel.create(config.embedding)
        self.embedding_fn = emb_model.embedding_fn()
        self.client = chromadb.Client(
            chromadb.config.Settings(
                # chroma_db_impl="duckdb+parquet",
                persist_directory=config.storage_path,
            )
        )
        if self.config.collection_name is not None:
            self.create_collection(
                self.config.collection_name,
                replace=self.config.replace_collection,
            )

    def clear_all_collections(self, really: bool = False, prefix: str = "") -> int:
        """Clear all collections in the vector store with the given prefix."""

        if not really:
            logger.warning("Not deleting all collections, set really=True to confirm")
            return 0
        coll = [c for c in self.client.list_collections() if c.name.startswith(prefix)]
        if len(coll) == 0:
            logger.warning(f"No collections found with prefix {prefix}")
            return 0
        n_empty_deletes = 0
        n_non_empty_deletes = 0
        for c in coll:
            n_empty_deletes += c.count() == 0
            n_non_empty_deletes += c.count() > 0
            self.client.delete_collection(name=c.name)
        logger.warning(
            f"""
            Deleted {n_empty_deletes} empty collections and 
            {n_non_empty_deletes} non-empty collections.
            """
        )
        return n_empty_deletes + n_non_empty_deletes

    def clear_empty_collections(self) -> int:
        colls = self.client.list_collections()
        n_deletes = 0
        for coll in colls:
            if coll.count() == 0:
                n_deletes += 1
                self.client.delete_collection(name=coll.name)
        return n_deletes

    def list_collections(self, empty: bool = False) -> List[str]:
        """
        List non-empty collections in the vector store.
        Args:
            empty (bool, optional): Whether to list empty collections.
        Returns:
            List[str]: List of non-empty collection names.
        """
        colls = self.client.list_collections()
        if empty:
            return [coll.name for coll in colls]
        return [coll.name for coll in colls if coll.count() > 0]

    def create_collection(self, collection_name: str, replace: bool = False) -> None:
        """
        Create a collection in the vector store, optionally replacing an existing
            collection if `replace` is True.
        Args:
            collection_name (str): Name of the collection to create or replace.
            replace (bool, optional): Whether to replace an existing collection.
                Defaults to False.

        """
        self.config.collection_name = collection_name
        if collection_name in self.list_collections(empty=True) and replace:
            logger.warning(f"Replacing existing collection {collection_name}")
            self.client.delete_collection(collection_name)
        self.collection = self.client.create_collection(
            name=self.config.collection_name,
            embedding_function=self.embedding_fn,
            get_or_create=not replace,
        )

    def add_documents(self, documents: Sequence[Document]) -> None:
        super().maybe_add_ids(documents)
        if documents is None:
            return
        contents: List[str] = [document.content for document in documents]
        # convert metadatas to dicts so chroma can handle them
        metadata_dicts: List[dict[str, Any]] = [
            d.metadata.dict_bool_int() for d in documents
        ]
        for m in metadata_dicts:
            # chroma does not handle non-atomic types in metadata
            m["window_ids"] = ",".join(m["window_ids"])

        ids = [str(d.id()) for d in documents]
        self.collection.add(
            # embedding_models=embedding_models,
            documents=contents,
            metadatas=metadata_dicts,
            ids=ids,
        )

    def get_all_documents(self, where: str = "") -> List[Document]:
        filter = json.loads(where) if where else None
        results = self.collection.get(
            include=["documents", "metadatas"],
            where=filter,
        )
        results["documents"] = [results["documents"]]
        results["metadatas"] = [results["metadatas"]]
        return self._docs_from_results(results)

    def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
        # get them one by one since chroma mangles the order of the results
        # when fetched from a list of ids.
        results = [
            self.collection.get(ids=[id], include=["documents", "metadatas"])
            for id in ids
        ]
        final_results = {}
        final_results["documents"] = [[r["documents"][0] for r in results]]
        final_results["metadatas"] = [[r["metadatas"][0] for r in results]]
        return self._docs_from_results(final_results)

    def delete_collection(self, collection_name: str) -> None:
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass

    def similar_texts_with_scores(
        self, text: str, k: int = 1, where: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        n = self.collection.count()
        filter = json.loads(where) if where else None
        results = self.collection.query(
            query_texts=[text],
            n_results=min(n, k),
            where=filter,
            include=["documents", "distances", "metadatas"],
        )
        docs = self._docs_from_results(results)
        # chroma distances are 1 - cosine.
        scores = [1 - s for s in results["distances"][0]]
        return list(zip(docs, scores))

    def _docs_from_results(self, results: Dict[str, Any]) -> List[Document]:
        """
        Helper function to convert results from ChromaDB to a list of Documents
        Args:
            results (dict): results from ChromaDB

        Returns:
            List[Document]: list of Documents
        """
        if len(results["documents"][0]) == 0:
            return []
        contents = results["documents"][0]
        if settings.debug:
            for i, c in enumerate(contents):
                print_long_text("red", "italic red", f"MATCH-{i}", c)
        metadatas = results["metadatas"][0]
        for m in metadatas:
            # restore the stringified list of window_ids into the original List[str]
            if m["window_ids"].strip() == "":
                m["window_ids"] = []
            else:
                m["window_ids"] = m["window_ids"].split(",")
        docs = [
            self.config.document_class(
                content=d, metadata=self.config.metadata_class(**m)
            )
            for d, m in zip(contents, metadatas)
        ]
        return docs
