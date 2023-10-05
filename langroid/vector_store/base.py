import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

from pydantic import BaseSettings

from langroid.embedding_models.base import EmbeddingModelsConfig
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.mytypes import Document
from langroid.utils.configuration import settings
from langroid.utils.output.printing import print_long_text

logger = logging.getLogger(__name__)


class VectorStoreConfig(BaseSettings):
    type: str = "qdrant"  # deprecated, keeping it for backward compatibility
    collection_name: str | None = None
    replace_collection: bool = False  # replace collection if it already exists
    storage_path: str = ".qdrant/data"
    cloud: bool = False
    batch_size: int = 200
    embedding: EmbeddingModelsConfig = OpenAIEmbeddingsConfig(
        model_type="openai",
    )
    timeout: int = 60
    host: str = "127.0.0.1"
    port: int = 6333
    # compose_file: str = "langroid/vector_store/docker-compose-qdrant.yml"


class VectorStore(ABC):
    """
    Abstract base class for a vector store.
    """

    def __init__(self, config: VectorStoreConfig):
        self.config = config

    @staticmethod
    def create(config: VectorStoreConfig) -> Optional["VectorStore"]:
        from langroid.vector_store.chromadb import ChromaDB, ChromaDBConfig
        from langroid.vector_store.qdrantdb import QdrantDB, QdrantDBConfig

        if isinstance(config, QdrantDBConfig):
            return QdrantDB(config)
        elif isinstance(config, ChromaDBConfig):
            return ChromaDB(config)
        else:
            logger.warning(
                f"""
                Unknown vector store config: {config.__repr_name__()},
                so skipping vector store creation!
                If you intended to use a vector-store, please set a specific 
                vector-store in your script, typically in the `vecdb` field of a 
                `ChatAgentConfig`, otherwise set it to None.
                """
            )
            return None

    @abstractmethod
    def clear_empty_collections(self) -> int:
        """Clear all empty collections in the vector store.
        Returns the number of collections deleted.
        """
        pass

    @abstractmethod
    def clear_all_collections(self, really: bool = False, prefix: str = "") -> int:
        """
        Clear all collections in the vector store.

        Args:
            really (bool, optional): Whether to really clear all collections.
                Defaults to False.
            prefix (str, optional): Prefix of collections to clear.
        Returns:
            int: Number of collections deleted.
        """
        pass

    @abstractmethod
    def list_collections(self, empty: bool = False) -> List[str]:
        """List all collections in the vector store
        (only non empty collections if empty=False).
        """
        pass

    def set_collection(self, collection_name: str, replace: bool = False) -> None:
        """
        Set the current collection to the given collection name.
        Args:
            collection_name (str): Name of the collection.
            replace (bool, optional): Whether to replace the collection if it
                already exists. Defaults to False.
        """

        self.config.collection_name = collection_name
        if collection_name not in self.list_collections() or replace:
            self.create_collection(collection_name, replace=replace)

    @abstractmethod
    def create_collection(self, collection_name: str, replace: bool = False) -> None:
        """Create a collection with the given name.
        Args:
            collection_name (str): Name of the collection.
            replace (bool, optional): Whether to replace the
                collection if it already exists. Defaults to False.
        """
        pass

    @abstractmethod
    def add_documents(self, documents: Sequence[Document]) -> None:
        pass

    @abstractmethod
    def similar_texts_with_scores(
        self,
        text: str,
        k: int = 1,
        where: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        pass

    @abstractmethod
    def get_all_documents(self) -> List[Document]:
        """
        Get all documents in the current collection.
        """
        pass

    @abstractmethod
    def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
        """
        Get documents by their ids.
        Args:
            ids (List[str]): List of document ids.

        Returns:
            List[Document]: List of documents
        """
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        pass

    def show_if_debug(self, doc_score_pairs: List[Tuple[Document, float]]) -> None:
        if settings.debug:
            for i, (d, s) in enumerate(doc_score_pairs):
                print_long_text("red", "italic red", f"\nMATCH-{i}\n", d.content)
