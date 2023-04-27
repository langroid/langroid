from llmagent.vector_store.base import VectorStore
from llmagent.mytypes import Document
from llmagent.utils.output.printing import print_long_text
from llmagent.embedding_models.models import embedding_model
from typing import List, Tuple
import chromadb

class ChromaDB(VectorStore):
    def __init__(self,
                 collection_name: str,
                 embedding_fn_type:str="openai",
                 storage_path: str = ".chromadb/data/",
                 ):
        super().__init__(collection_name)
        emb_model = embedding_model(embedding_fn_type)
        self.embedding_fn: EmbeddingFunction = emb_model.embedding_fn()
        self.client = chromadb.Client(chromadb.config.Settings(
            #chroma_db_impl="duckdb+parquet",
            persist_directory=storage_path,
        ))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            # metadata={
            #     "hnsw:space": "cosine",
            #     "hnsw:construction_ef": 9,
            #     "hnsw:M": 4,
            #     "hnsw:search_ef": 4,
            # }
        )

    @classmethod
    def from_documents(cls,
                       collection_name:str,
                       documents: List[Document],
                       storage_path: str = ".chromadb/data/",
                       embedding_fn_type:str ="openai",
                       embedding_model:str ="text-embedding-ada-002",
                       embeddings=None,
                       ):
        instance = cls(
            collection_name=collection_name,
            embedding_fn_type=embedding_fn_type,
            storage_path=storage_path
        )

        instance.add_documents(
            embeddings=embeddings, documents=documents,
        )
        return instance

    def add_documents(self, documents:List[Document]=None):
        contents: List[str]  = [document.content for document in documents]
        metadatas: dict = [document.metadata for document in documents]
        ids = range(len(documents))
        ids = ["id" + str(id) for id in ids]
        self.collection.add(
            #embedding_models=embedding_models,
            documents=contents,
            metadatas=metadatas,
            ids=ids,
        )

    def similar_texts_with_scores(
            self,
            text:str, k:int=None,
            where:str=None, debug:bool=False
    ) -> List[Tuple[Document, float]]:
        results = self.collection.query(
            query_texts=[text],
            n_results=k,
            where=where,
            include=["documents", "distances", "metadatas"],
        )
        contents = results["documents"][0]
        if debug:
            for i, c in enumerate(contents):
                print_long_text("red", "italic red", f"MATCH-{i}", c)
        metadatas = results["metadatas"][0]
        docs = [Document(content=d, metadata=m) for d, m in zip(contents, metadatas)]
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