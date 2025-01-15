import logging
from typing import List, Optional, Sequence, Tuple

import psycopg2
from pgvector.psycopg2 import register_vector

from langroid.embedding_models.base import EmbeddingModel
from langroid.mytypes import Document
from langroid.vector_store.base import VectorStore, VectorStoreConfig

logger = logging.getLogger(__name__)


class PGVectorConfig(VectorStoreConfig):
    """
    Configuration for PGVector vector store.
    """

    table_name: str = "vectors"
    distance_metric: str = "cosine"
    database: str
    user: str
    password: str
    host: str = "localhost"
    port: int = 5432


class PGVector(VectorStore):
    """
    Implementation of a vector store using pgvector with PostgreSQL.
    """

    def __init__(self, config: PGVectorConfig):
        super().__init__(config)
        self.config: PGVectorConfig = config

        # Connect to PostgreSQL
        self.conn = psycopg2.connect(
            host=config.host,
            port=config.port,
            database=config.database,
            user=config.user,
            password=config.password,
        )
        register_vector(self.conn)
        self.cursor = self.conn.cursor()

        # Ensure the table exists
        if not config.collection_name:
            raise ValueError("Collection name must be provided.")
        self.create_collection(
            config.collection_name, replace=config.replace_collection
        )

    def clear_empty_collections(self) -> int:
        """Clear empty collections (not applicable to pgvector)."""
        logger.warning("`clear_empty_collections` not applicable to pgvector.")
        return 0

    def clear_all_collections(self, really: bool = False, prefix: str = "") -> int:
        """
        Clear all collections by deleting the vectors table if confirmed.

        Args:
            really (bool, optional): Whether to confirm deletion. Defaults to False.
            prefix (str, optional): Prefix of collections to clear (unused here).

        Returns:
            int: Number of collections deleted.
        """
        if not really:
            logger.warning("Set `really=True` to confirm deletion of all collections.")
            return 0

        self.cursor.execute(f"DROP TABLE IF EXISTS {self.config.table_name};")
        self.conn.commit()
        logger.info(f"Cleared table: {self.config.table_name}.")
        return 1

    def list_collections(self, empty: bool = False) -> List[str]:
        """
        List all collections (always one for pgvector).

        Args:
            empty (bool, optional): Whether to list empty collections (unused).

        Returns:
            List[str]: List containing the single table name.
        """
        return [self.config.table_name]

    def set_collection(self, collection_name: str, replace: bool = False) -> None:
        """
        Set the current collection (table) to the given name.

        Args:
            collection_name (str): Name of the table.
            replace (bool, optional): Whether to replace the table if it exists.
        """
        self.config.collection_name = collection_name
        if replace:
            self.create_collection(collection_name, replace=True)

    def create_collection(self, collection_name: str, replace: bool = False) -> None:
        """
        Create a collection (table) in the database.

        Args:
            collection_name (str): Name of the table to create.
            replace (bool, optional): Whether to replace the table if it exists.
        """
        if replace:
            self.cursor.execute(f"DROP TABLE IF EXISTS {collection_name};")
        self.cursor.execute(
            f"""
        CREATE TABLE IF NOT EXISTS {collection_name} (
            id SERIAL PRIMARY KEY,
            vector VECTOR({self.embedding_model.embedding_dims}),
            metadata JSONB
        );
        """
        )
        self.conn.commit()
        logger.info(f"Table {collection_name} created or exists.")

    def add_documents(self, documents: Sequence[Document]) -> None:
        """
        Add documents to the vector store.

        Args:
            documents (Sequence[Document]): Documents to add.
        """
        super().maybe_add_ids(documents)
        for doc in documents:
            vector = self.embedding_model.embedding_fn()([doc.content])[0]
            metadata = doc.dict()
            self.cursor.execute(
                f"""
            INSERT INTO {self.config.collection_name} (vector, metadata)
            VALUES (%s, %s);
            """,
                (vector, metadata),
            )
        self.conn.commit()

    def similar_texts_with_scores(
        self,
        text: str,
        k: int = 1,
        where: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Find k most similar texts to the given input text.

        Args:
            text (str): Input text to search for.
            k (int, optional): Number of similar texts to retrieve. Defaults to 1.
            where (Optional[str], optional): Not implemented for pgvector.

        Returns:
            List[Tuple[Document, float]]: List of (Document, score) tuples.
        """
        query_vector = self.embedding_model.embedding_fn()([text])[0]
        self.cursor.execute(
            f"""
        SELECT metadata, (vector <-> %s) AS distance
        FROM {self.config.collection_name}
        ORDER BY distance ASC
        LIMIT %s;
        """,
            (query_vector, k),
        )
        rows = self.cursor.fetchall()
        return [(Document(**row[0]), row[1]) for row in rows]

    def get_all_documents(self, where: str = "") -> List[Document]:
        """
        Retrieve all documents from the collection.

        Args:
            where (str, optional): Not implemented for pgvector.

        Returns:
            List[Document]: All documents in the collection.
        """
        self.cursor.execute(f"SELECT metadata FROM {self.config.collection_name};")
        rows = self.cursor.fetchall()
        return [Document(**row[0]) for row in rows]

    def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
        """
        Retrieve documents by IDs.

        Args:
            ids (List[str]): List of document IDs.

        Returns:
            List[Document]: Matching documents.
        """
        self.cursor.execute(
            f"""
        SELECT metadata FROM {self.config.collection_name}
        WHERE id = ANY(%s);
        """,
            (ids,),
        )
        rows = self.cursor.fetchall()
        return [Document(**row[0]) for row in rows]

    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection (table) from the database.

        Args:
            collection_name (str): Name of the table to delete.
        """
        self.cursor.execute(f"DROP TABLE IF EXISTS {collection_name};")
        self.conn.commit()
        logger.info(f"Table {collection_name} deleted.")
