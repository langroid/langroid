from abc import abstractmethod
from dataclasses import dataclass
from llmagent.vector_store.config import VectorStoreConfig
from llmagent.utils.output.printing import print_long_text
from llmagent.utils.configuration import settings
import logging
logger = logging.getLogger(__name__)

@dataclass
class VectorStore(VectorStoreConfig):
    def create(self):
        from llmagent.vector_store.qdrantdb import QdrantDB
        from llmagent.vector_store.faissdb import FAISSDB
        from llmagent.vector_store.chromadb import ChromaDB
        vecstore_class = dict(faiss = FAISSDB, qdrant = QdrantDB, chroma = ChromaDB).get(
            self.type, ChromaDB
        )

        return vecstore_class(
            collection_name=self.collection_name,
            embedding_fn_type=self.embedding_fn_type,
            host=self.host,
            port=self.port,
        )



    @abstractmethod
    def from_documents(self, collection_name, documents, embeddings=None,
                       storage_path=None,
                       metadatas=None, ids=None):
        pass

    @abstractmethod
    def add_documents(self, embeddings=None, documents=None,
                      metadatas=None, ids=None):
        pass

    @abstractmethod
    def similar_texts_with_scores(self, text:str, k:int=None,
                                  where:str=None, debug:bool=False):
        pass

    def show_if_debug(self, doc_score_pairs):
        if settings.debug:
            for i, (d, s) in enumerate(doc_score_pairs):
                print_long_text("red", "italic red", f"MATCH-{i}", d.content)
