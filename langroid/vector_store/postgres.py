import hashlib
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langroid.embedding_models.base import (
    EmbeddingModelsConfig,
)
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.exceptions import LangroidImportError
from langroid.mytypes import DocMetaData, Document
from langroid.vector_store.base import VectorStore, VectorStoreConfig

has_postgres: bool = True
try:
    from sqlalchemy import (
        Column,
        MetaData,
        String,
        Table,
        case,
        create_engine,
        inspect,
        text,
    )
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.engine import Connection, Engine
    from sqlalchemy.sql.expression import insert
except ImportError:
    Engine = Any  # type: ignore
    Connection = Any  # type: ignore
    has_postgres = False

logger = logging.getLogger(__name__)


class PostgresDBConfig(VectorStoreConfig):
    collection_name: str = "embeddings"
    cloud: bool = False
    docker: bool = True
    host: str = "127.0.0.1"
    port: int = 5432
    replace_collection: bool = False
    embedding: EmbeddingModelsConfig = OpenAIEmbeddingsConfig()
    pool_size: int = 10
    max_overflow: int = 20
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200


class PostgresDB(VectorStore):
    def __init__(self, config: PostgresDBConfig = PostgresDBConfig()):
        super().__init__(config)
        if not has_postgres:
            raise LangroidImportError("pgvector", "postgres")
        try:
            from sqlalchemy.orm import sessionmaker
        except ImportError:
            raise LangroidImportError("sqlalchemy", "postgres")

        self.config: PostgresDBConfig = config
        self.engine = self._create_engine()
        PostgresDB._create_vector_extension(self.engine)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )
        self.metadata = MetaData()
        self._setup_table()

    def _create_engine(self) -> Engine:
        """Creates a SQLAlchemy engine based on the configuration."""

        connection_string: str | None = None  # Ensure variable is always defined

        if self.config.cloud:
            connection_string = os.getenv("POSTGRES_CONNECTION_STRING")

            if connection_string and connection_string.startswith("postgres://"):
                connection_string = connection_string.replace(
                    "postgres://", "postgresql+psycopg2://", 1
                )
            elif not connection_string:
                raise ValueError("Provide the POSTGRES_CONNECTION_STRING.")

        elif self.config.docker:
            username = os.getenv("POSTGRES_USER", "postgres")
            password = os.getenv("POSTGRES_PASSWORD", "postgres")
            database = os.getenv("POSTGRES_DB", "langroid")

            if not (username and password and database):
                raise ValueError(
                    "Provide POSTGRES_USER, POSTGRES_PASSWORD, " "POSTGRES_DB. "
                )

            connection_string = (
                f"postgresql+psycopg2://{username}:{password}@"
                f"{self.config.host}:{self.config.port}/{database}"
            )
            self.config.cloud = False  # Ensures cloud is disabled if using Docker

        else:
            raise ValueError(
                "Provide either Docker or Cloud config to connect to the database."
            )

        return create_engine(
            connection_string,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
        )

    def _setup_table(self) -> None:
        try:
            from pgvector.sqlalchemy import Vector
        except ImportError as e:
            raise LangroidImportError(extra="postgres", error=str(e))

        if self.config.replace_collection:
            self.delete_collection(self.config.collection_name)

        self.embeddings_table = Table(
            self.config.collection_name,
            self.metadata,
            Column("id", String, primary_key=True, nullable=False, unique=True),
            Column("embedding", Vector(self.embedding_dim)),
            Column("document", String),
            Column("cmetadata", JSONB),
            extend_existing=True,
        )

        self.metadata.create_all(self.engine)
        self.metadata.reflect(bind=self.engine, only=[self.config.collection_name])

        # Create HNSW index for embeddings column if it doesn't exist.
        # This index enables efficient nearest-neighbor search using cosine similarity.
        # PostgreSQL automatically builds the index after creation;
        # no manual step required.
        # Read more about pgvector hnsw index here:
        # https://github.com/pgvector/pgvector?tab=readme-ov-file#hnsw

        index_name = f"hnsw_index_{self.config.collection_name}_embedding"
        with self.engine.connect() as connection:
            if not self.index_exists(connection, index_name):
                connection.execute(text("COMMIT"))
                create_index_query = text(
                    f"""
                    CREATE INDEX CONCURRENTLY IF NOT EXISTS {index_name}
                    ON {self.config.collection_name}
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (
                        m = {self.config.hnsw_m},
                        ef_construction = {self.config.hnsw_ef_construction}
                    );
                    """
                )
                connection.execute(create_index_query)

    def index_exists(self, connection: Connection, index_name: str) -> bool:
        """Check if an index exists."""
        query = text(
            "SELECT 1 FROM pg_indexes WHERE indexname = :index_name"
        ).bindparams(index_name=index_name)
        result = connection.execute(query).scalar()
        return bool(result)

    @staticmethod
    def _create_vector_extension(conn: Engine) -> None:

        with conn.connect() as connection:
            with connection.begin():
                # The number is a unique identifier used to lock a specific resource
                # during transaction. Any 64-bit integer can be used for advisory locks.
                # Acquire advisory lock to ensure atomic, isolated setup
                # and prevent race conditions.

                statement = text(
                    "SELECT pg_advisory_xact_lock(1573678846307946496);"
                    "CREATE EXTENSION IF NOT EXISTS vector;"
                )
                connection.execute(statement)

    def set_collection(self, collection_name: str, replace: bool = False) -> None:
        inspector = inspect(self.engine)
        table_exists = collection_name in inspector.get_table_names()

        if (
            collection_name == self.config.collection_name
            and table_exists
            and not replace
        ):
            return
        else:
            self.config.collection_name = collection_name
            self.config.replace_collection = replace
            self._setup_table()

    def list_collections(self, empty: bool = True) -> List[str]:
        inspector = inspect(self.engine)
        table_names = inspector.get_table_names()

        with self.SessionLocal() as session:
            collections = []
            for table_name in table_names:
                table = Table(table_name, self.metadata, autoload_with=self.engine)
                if empty:
                    collections.append(table_name)
                else:
                    # Efficiently check for non-emptiness
                    if session.query(table.select().limit(1).exists()).scalar():
                        collections.append(table_name)
            return collections

    def create_collection(self, collection_name: str, replace: bool = False) -> None:
        self.set_collection(collection_name, replace=replace)

    def delete_collection(self, collection_name: str) -> None:
        """
        Deletes a collection and its associated HNSW index, handling metadata
        synchronization issues.
        """
        with self.engine.connect() as connection:
            connection.execute(text("COMMIT"))
            index_name = f"hnsw_index_{collection_name}_embedding"
            drop_index_query = text(f"DROP INDEX CONCURRENTLY IF EXISTS {index_name}")
            connection.execute(drop_index_query)

            # 3. Now, drop the table using SQLAlchemy
            table = Table(collection_name, self.metadata)
            table.drop(self.engine, checkfirst=True)

            # 4. Refresh metadata again after dropping the table
            self.metadata.clear()
            self.metadata.reflect(bind=self.engine)

    def clear_all_collections(self, really: bool = False, prefix: str = "") -> int:
        if not really:
            logger.warning("Not deleting all tables, set really=True to confirm")
            return 0

        inspector = inspect(self.engine)
        table_names = inspector.get_table_names()

        with self.SessionLocal() as session:
            deleted_count = 0
            for table_name in table_names:
                if table_name.startswith(prefix):
                    # Use delete_collection to handle index and table deletion
                    self.delete_collection(table_name)
                    deleted_count += 1
            session.commit()
            logger.warning(f"Deleted {deleted_count} tables with prefix '{prefix}'.")
            return deleted_count

    def clear_empty_collections(self) -> int:
        inspector = inspect(self.engine)
        table_names = inspector.get_table_names()

        with self.SessionLocal() as session:
            deleted_count = 0
            for table_name in table_names:
                table = Table(table_name, self.metadata, autoload_with=self.engine)

                # Efficiently check for emptiness without fetching all rows
                if session.query(table.select().limit(1).exists()).scalar():
                    continue

                # Use delete_collection to handle index and table deletion
                self.delete_collection(table_name)
                deleted_count += 1

            session.commit()  # Commit is likely not needed here
            logger.warning(f"Deleted {deleted_count} empty tables.")
            return deleted_count

    def _parse_embedding_store_record(self, res: Any) -> Dict[str, Any]:
        metadata = res.cmetadata or {}
        metadata["id"] = res.id
        return {
            "content": res.document,
            "metadata": DocMetaData(**metadata),
        }

    def get_all_documents(self, where: str = "") -> List[Document]:
        with self.SessionLocal() as session:
            query = session.query(self.embeddings_table)

            # Apply 'where' clause if provided
            if where:
                try:
                    where_json = json.loads(where)
                    query = query.filter(
                        self.embeddings_table.c.cmetadata.contains(where_json)
                    )
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in 'where' clause: {where}")
                    return []  # Return empty list or handle error as appropriate

            results = query.all()
            documents = [
                Document(**self._parse_embedding_store_record(res)) for res in results
            ]
            return documents

    def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
        with self.SessionLocal() as session:
            # Add a CASE statement to preserve the order of IDs
            case_stmt = case(
                {id_: index for index, id_ in enumerate(ids)},
                value=self.embeddings_table.c.id,
            )

            query = (
                session.query(self.embeddings_table)
                .filter(self.embeddings_table.c.id.in_(ids))
                .order_by(case_stmt)  # Order by the CASE statement
            )
            results = query.all()

            documents = [
                Document(**self._parse_embedding_store_record(row)) for row in results
            ]
            return documents

    def add_documents(self, documents: Sequence[Document]) -> None:
        super().maybe_add_ids(documents)
        for doc in documents:
            doc.metadata.id = str(PostgresDB._id_to_uuid(doc.metadata.id, doc.metadata))

        embeddings = self.embedding_fn([doc.content for doc in documents])

        batch_size = self.config.batch_size
        with self.SessionLocal() as session:
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i : i + batch_size]
                batch_embeddings = embeddings[i : i + batch_size]

                new_records = [
                    {
                        "id": doc.metadata.id,
                        "embedding": embedding,
                        "document": doc.content,
                        "cmetadata": doc.metadata.dict(),
                    }
                    for doc, embedding in zip(batch_docs, batch_embeddings)
                ]

                if new_records:
                    stmt = insert(self.embeddings_table).values(new_records)
                    session.execute(stmt)
                session.commit()

    @staticmethod
    def _id_to_uuid(id: str, obj: object) -> str:
        try:
            doc_id = str(uuid.UUID(id))
        except ValueError:
            obj_repr = repr(obj)

            obj_hash = hashlib.sha256(obj_repr.encode()).hexdigest()

            combined = f"{id}-{obj_hash}"

            doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, combined))

        return doc_id

    def similar_texts_with_scores(
        self,
        query: str,
        k: int = 1,
        where: Optional[str] = None,
        neighbors: int = 1,  # Parameter not used in this implementation
    ) -> List[Tuple[Document, float]]:
        embedding = self.embedding_fn([query])[0]

        with self.SessionLocal() as session:
            # Calculate the score (1 - cosine_distance) and label it as "score"
            score = (
                1 - (self.embeddings_table.c.embedding.cosine_distance(embedding))
            ).label("score")

            if where is not None:
                try:
                    json_query = json.loads(where)
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON in 'where' clause: {where}")

                results = (
                    session.query(
                        self.embeddings_table.c.id,
                        self.embeddings_table.c.document,
                        self.embeddings_table.c.cmetadata,
                        score,  # Select the calculated score
                    )
                    .filter(self.embeddings_table.c.cmetadata.contains(json_query))
                    .order_by(score.desc())  # Order by score in descending order
                    .limit(k)
                    .all()
                )
            else:
                results = (
                    session.query(
                        self.embeddings_table.c.id,
                        self.embeddings_table.c.document,
                        self.embeddings_table.c.cmetadata,
                        score,  # Select the calculated score
                    )
                    .order_by(score.desc())  # Order by score in descending order
                    .limit(k)
                    .all()
                )

            documents_with_scores = [
                (
                    Document(
                        content=result.document,
                        metadata=DocMetaData(**(result.cmetadata or {})),
                    ),
                    result.score,  # Use the score from the query result
                )
                for result in results
            ]

            return documents_with_scores
