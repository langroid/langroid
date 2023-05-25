from pydantic import BaseSettings
from abc import ABC, abstractmethod
import logging
from llmagent.embedding_models.base import EmbeddingModelsConfig
from llmagent.utils.output.printing import print_long_text
from llmagent.utils.configuration import settings
from llmagent.mytypes import Document
import uuid
import hashlib

logger = logging.getLogger(__name__)


class VectorStoreConfig(BaseSettings):
    collection_name: str = "default"
    storage_path: str = ".qdrant/data"
    cloud: bool = False
    embedding: EmbeddingModelsConfig = EmbeddingModelsConfig(
        model_type="openai",
    )
    type: str = "qdrant"
    host: str = "127.0.0.1"
    port: int = 6333
    # compose_file: str = "llmagent/vector_store/docker-compose-qdrant.yml"


class VectorStore(ABC):
    @staticmethod
    def create(config: VectorStoreConfig):
        from llmagent.vector_store.qdrantdb import QdrantDB
        from llmagent.vector_store.chromadb import ChromaDB

        vecstore_class = dict(qdrant=QdrantDB, chroma=ChromaDB).get(
            config.type, QdrantDB
        )

        return vecstore_class(config)

    # @abstractmethod
    # def from_documents(self, collection_name, documents, embeddings=None,
    #                    storage_path=None,
    #                    metadatas=None, ids=None):
    #     pass

    @abstractmethod
    def add_documents(self, embeddings=None, documents=None, metadatas=None, ids=None):
        pass

    @abstractmethod
    def similar_texts_with_scores(
        self, text: str, k: int = 1, where: str = None, debug: bool = False
    ):
        pass

    @staticmethod
    def _unique_hash_id(doc: Document) -> str:
        # Encode the document as UTF-8
        doc_utf8 = str(doc).encode("utf-8")

        # Create a SHA256 hash object
        sha256_hash = hashlib.sha256()

        # Update the hash object with the bytes of the document
        sha256_hash.update(doc_utf8)

        # Get the hexadecimal representation of the hash
        hash_hex = sha256_hash.hexdigest()

        # Convert the first part of the hash to a UUID
        hash_uuid = uuid.UUID(hash_hex[:32])

        return str(hash_uuid)

    @abstractmethod
    def delete_collection(self, collection_name: str):
        pass

    def show_if_debug(self, doc_score_pairs):
        if settings.debug:
            for i, (d, s) in enumerate(doc_score_pairs):
                print_long_text("red", "italic red", f"MATCH-{i}", d.content)
