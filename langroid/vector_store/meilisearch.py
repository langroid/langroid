"""
MeiliSearch as a pure document store, without its
(experimental) vector-store functionality.
We aim to use MeiliSearch for fast lexical search.
Note that what we call "Collection" in Langroid is referred to as
"Index" in MeiliSearch. Each data-store has its own terminology,
but for uniformity we use the Langroid terminology here.
"""

import asyncio
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import meilisearch_python_sdk as meilisearch
from dotenv import load_dotenv
from meilisearch_python_sdk.index import AsyncIndex
from meilisearch_python_sdk.models.documents import DocumentsInfo

from langroid.mytypes import DocMetaData, Document
from langroid.utils.configuration import settings
from langroid.vector_store.base import VectorStore, VectorStoreConfig

logger = logging.getLogger(__name__)


class MeiliSearchConfig(VectorStoreConfig):
    cloud: bool = False
    collection_name: str | None = None
    primary_key: str = "id"
    port = 7700


class MeiliSearch(VectorStore):
    def __init__(self, config: MeiliSearchConfig = MeiliSearchConfig()):
        super().__init__(config)
        self.config: MeiliSearchConfig = config
        self.host = config.host
        self.port = config.port
        load_dotenv()
        self.key = os.getenv("MEILISEARCH_API_KEY") or "masterKey"
        self.url = os.getenv("MEILISEARCH_API_URL") or f"http://{self.host}:{self.port}"
        if config.cloud and None in [self.key, self.url]:
            logger.warning(
                f"""MEILISEARCH_API_KEY, MEILISEARCH_API_URL env variable must be set 
                to use MeiliSearch in cloud mode. Please set these values 
                in your .env file. Switching to local MeiliSearch at 
                {self.url} 
                """
            )
            config.cloud = False

        self.client: Callable[[], meilisearch.AsyncClient] = lambda: (
            meilisearch.AsyncClient(url=self.url, api_key=self.key)
        )

        # Note: Only create collection if a non-null collection name is provided.
        # This is useful to delay creation of db until we have a suitable
        # collection name (e.g. we could get it from the url or folder path).
        if config.collection_name is not None:
            self.create_collection(
                config.collection_name, replace=config.replace_collection
            )

    def clear_empty_collections(self) -> int:
        """All collections are treated as non-empty in MeiliSearch, so this is a
        no-op"""
        return 0

    async def _async_delete_indices(self, uids: List[str]) -> List[bool]:
        """Delete any indicecs in `uids` that exist.
        Returns list of bools indicating whether the index has been deleted"""
        async with self.client() as client:
            result = await asyncio.gather(
                *[client.delete_index_if_exists(uid=uid) for uid in uids]
            )
        return result

    def clear_all_collections(self, really: bool = False, prefix: str = "") -> int:
        """Delete all indices whose names start with `prefix`"""
        if not really:
            logger.warning("Not deleting all collections, set really=True to confirm")
            return 0
        coll_names = [c for c in self.list_collections() if c.startswith(prefix)]
        deletes = asyncio.run(self._async_delete_indices(coll_names))
        n_deletes = sum(deletes)
        logger.warning(f"Deleted {n_deletes} indices in MeiliSearch")
        return n_deletes

    def _list_all_collections(self) -> List[str]:
        """
        List all collections, including empty ones.
        Returns:
            List of collection names.
        """
        return self.list_collections()

    async def _async_get_indexes(self) -> List[AsyncIndex]:
        async with self.client() as client:
            indexes = await client.get_indexes(limit=10_000)
        return [] if indexes is None else indexes

    async def _async_get_index(self, index_uid: str) -> AsyncIndex:
        async with self.client() as client:
            index = await client.get_index(index_uid)
        return index

    def list_collections(self, empty: bool = False) -> List[str]:
        """
        Returns:
            List of index names stored. We treat any existing index as non-empty.
        """
        indexes = asyncio.run(self._async_get_indexes())
        if len(indexes) == 0:
            return []
        else:
            return [ind.uid for ind in indexes]

    async def _async_create_index(self, collection_name: str) -> AsyncIndex:
        async with self.client() as client:
            index = await client.create_index(
                uid=collection_name,
                primary_key=self.config.primary_key,
            )
        return index

    async def _async_delete_index(self, collection_name: str) -> bool:
        """Delete index if it exists. Returns True iff index was deleted"""
        async with self.client() as client:
            result = await client.delete_index_if_exists(uid=collection_name)
        return result

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
        collections = self.list_collections()
        if collection_name in collections:
            logger.warning(
                f"MeiliSearch Non-empty Index {collection_name} already exists"
            )
            if not replace:
                logger.warning("Not replacing collection")
                return
            else:
                logger.warning("Recreating fresh collection")
                asyncio.run(self._async_delete_index(collection_name))
        asyncio.run(self._async_create_index(collection_name))
        collection_info = asyncio.run(self._async_get_index(collection_name))
        if settings.debug:
            level = logger.getEffectiveLevel()
            logger.setLevel(logging.INFO)
            logger.info(collection_info)
            logger.setLevel(level)

    async def _async_add_documents(
        self, collection_name: str, documents: Sequence[Dict[str, Any]]
    ) -> None:
        async with self.client() as client:
            index = client.index(collection_name)
            await index.add_documents_in_batches(
                documents=documents,
                batch_size=self.config.batch_size,
                primary_key=self.config.primary_key,
            )

    def add_documents(self, documents: Sequence[Document]) -> None:
        super().maybe_add_ids(documents)
        if len(documents) == 0:
            return
        colls = self._list_all_collections()
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot ingest docs")
        if self.config.collection_name not in colls:
            self.create_collection(self.config.collection_name, replace=True)
        docs = [
            dict(
                id=d.id(),
                content=d.content,
                metadata=d.metadata.dict(),
            )
            for d in documents
        ]
        asyncio.run(self._async_add_documents(self.config.collection_name, docs))

    def delete_collection(self, collection_name: str) -> None:
        asyncio.run(self._async_delete_index(collection_name))

    def _to_int_or_uuid(self, id: str) -> int | str:
        try:
            return int(id)
        except ValueError:
            return id

    async def _async_get_documents(self, where: str = "") -> DocumentsInfo:
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot retrieve docs")
        filter = [] if where is None else where
        async with self.client() as client:
            index = client.index(self.config.collection_name)
            documents = await index.get_documents(limit=10_000, filter=filter)
        return documents

    def get_all_documents(self, where: str = "") -> List[Document]:
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot retrieve docs")
        docs = asyncio.run(self._async_get_documents(where))
        if docs is None:
            return []
        doc_results = docs.results
        return [
            Document(
                content=d["content"],
                metadata=DocMetaData(**d["metadata"]),
            )
            for d in doc_results
        ]

    async def _async_get_documents_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot retrieve docs")
        async with self.client() as client:
            index = client.index(self.config.collection_name)
            documents = await asyncio.gather(*[index.get_document(id) for id in ids])
        return documents

    def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot retrieve docs")
        docs = asyncio.run(self._async_get_documents_by_ids(ids))
        return [
            Document(
                content=d["content"],
                metadata=DocMetaData(**d["metadata"]),
            )
            for d in docs
        ]

    async def _async_search(
        self,
        query: str,
        k: int = 20,
        filter: str | list[str | list[str]] | None = None,
    ) -> List[Dict[str, Any]]:
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot search")
        async with self.client() as client:
            index = client.index(self.config.collection_name)
            results = await index.search(
                query,
                limit=k,
                show_ranking_score=True,
                filter=filter,
            )
        return results.hits

    def similar_texts_with_scores(
        self,
        text: str,
        k: int = 20,
        where: Optional[str] = None,
        neighbors: int = 0,  # ignored
    ) -> List[Tuple[Document, float]]:
        filter = [] if where is None else where
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot search")
        _docs = asyncio.run(self._async_search(text, k, filter))  # type: ignore
        if len(_docs) == 0:
            logger.warning(f"No matches found for {text}")
            return []
        scores = [h["_rankingScore"] for h in _docs]
        if settings.debug:
            logger.info(f"Found {len(_docs)} matches, max score: {max(scores)}")
        docs = [
            Document(
                content=d["content"],
                metadata=DocMetaData(**d["metadata"]),
            )
            for d in _docs
        ]
        doc_score_pairs = list(zip(docs, scores))
        self.show_if_debug(doc_score_pairs)
        return doc_score_pairs
