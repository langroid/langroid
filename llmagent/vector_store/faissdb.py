from llmagent.vector_store.base import VectorStore
from llmagent.mytypes import Document
from langchain.schema import Document as LDocument
from typing import List
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings


class FAISSDB(VectorStore):
    def __init__(
        self,
        collection_name: str,
        embedding_fn_type: str = "openai",
        storage_path: str = ".faissdb/data/",
        embedding_model: str = "text-embedding-ada-002",
    ):
        super().__init__(collection_name)

        assert (
            embedding_fn_type == "openai"
        ), "FAISSDB only supports OpenAI embedding function"
        self.embedding_fn = OpenAIEmbeddings()
        self.collection: FAISS = FAISS

    @classmethod
    def from_documents(
        cls,
        collection_name: str,
        documents: List[Document],
        storage_path: str = ".faissdb/data/",
        embedding_fn_type: str = "openai",
        embedding_model: str = "text-embedding-ada-002",
        embeddings=None,
    ):
        instance = cls(
            collection_name=collection_name,
            storage_path=storage_path,
            embedding_fn_type=embedding_fn_type,
            embedding_model=embedding_model,
        )
        lc_docs = [
            LDocument(page_content=d.content, metadata=d.metadata) for d in documents
        ]
        instance.collection = instance.collection.from_documents(
            lc_docs, instance.embedding_fn
        )
        return instance

    def add_documents(self, documents: List[Document] = None):
        raise NotImplementedError

    def similar_texts_with_scores(self, text: str, k: int = None, where: str = None):
        doc_score_pairs = self.collection.similarity_search_with_score(text, k=k)
        # convert langchain docs to our docs
        doc_score_pairs = [
            (Document(content=d.page_content, metadata=d.metadata), s)
            for d, s in doc_score_pairs
        ]
        self.show_if_debug(doc_score_pairs)
        return doc_score_pairs


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
