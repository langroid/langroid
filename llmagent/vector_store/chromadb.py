import logging
from typing import Any, List, Optional, Tuple

import chromadb

from llmagent.embedding_models.base import (
    EmbeddingModel,
    EmbeddingModelsConfig,
)
from llmagent.mytypes import DocMetaData, Document
from llmagent.utils.configuration import settings
from llmagent.utils.output.printing import print_long_text
from llmagent.vector_store.base import VectorStore, VectorStoreConfig

logger = logging.getLogger(__name__)


class ChromaDBConfig(VectorStoreConfig):
    type: str = "chroma"
    collection_name: str = "chroma-llmagent"
    storage_path: str = ".chroma/data"
    embedding: EmbeddingModelsConfig = EmbeddingModelsConfig(
        model_type="openai",
    )
    host: str = "127.0.0.1"
    port: int = 6333


class ChromaDB(VectorStore):
    def __init__(self, config: ChromaDBConfig):
        super().__init__()
        self.config = config
        emb_model = EmbeddingModel.create(config.embedding)
        self.embedding_fn = emb_model.embedding_fn()
        self.client = chromadb.Client(
            chromadb.config.Settings(
                # chroma_db_impl="duckdb+parquet",
                persist_directory=config.storage_path,
            )
        )
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            embedding_function=self.embedding_fn,
            # metadata={
            #     "hnsw:space": "cosine",
            #     "hnsw:construction_ef": 9,
            #     "hnsw:M": 4,
            #     "hnsw:search_ef": 4,
            # }
        )

    def add_documents(self, documents: Optional[List[Document]] = None) -> None:
        if documents is None:
            return
        contents: List[str] = [document.content for document in documents]
        metadatas: List[dict[str, Any]] = [
            document.metadata.dict() for document in documents
        ]
        ids = range(len(documents))
        ids_str = ["id" + str(id) for id in ids]
        self.collection.add(
            # embedding_models=embedding_models,
            documents=contents,
            metadatas=metadatas,
            ids=ids_str,
        )

    def delete_collection(self, collection_name: str) -> None:
        self.client.delete_collection(name=collection_name)

    def similar_texts_with_scores(
        self, text: str, k: int = 1, where: Optional[str] = None
    ) -> Optional[List[Tuple[Document, float]]]:
        results = self.collection.query(
            query_texts=[text],
            n_results=k,
            where=where,
            include=["documents", "distances", "metadatas"],
        )
        if len(results["documents"]) == 0:
            return None
        contents = results["documents"][0]
        if settings.debug:
            for i, c in enumerate(contents):
                print_long_text("red", "italic red", f"MATCH-{i}", c)
        metadatas = results["metadatas"][0]
        docs = [
            Document(content=d, metadata=DocMetaData.parse_obj(m))
            for d, m in zip(contents, metadatas)
        ]
        scores = results["distances"][0]
        return list(zip(docs, scores))


# Example usage and testing
# chroma_db = ChromaDB.from_documents(
#     collection_name="all-my-documents",
#     documents=["doc1000101", "doc288822"],
#     metadatas=[{"style": "style1"}, {"style": "style2"}],
#     ids=["uri9", "uri10"]
# )
# results = chroma_db.query(
#     query_texts=["This is a query document"],
#     n_results=2
# )
# print(results)
