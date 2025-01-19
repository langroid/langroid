import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Index, MetaData, String, create_engine, inspect, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker

from langroid.embedding_models.base import EmbeddingModelsConfig
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.mytypes import Document
from langroid.vector_store.base import VectorStore, VectorStoreConfig

Base = declarative_base()
logger = logging.getLogger(__name__)


class PGVectorConfig(VectorStoreConfig):
    docker: bool = True
    collection_name: str | None = "temp"
    storage_path: str = ".postgres/data"
    host: str = "localhost"
    port: int = 5435
    username: str = "postgres"
    password: str = "postgres"
    database: str = "langroid"
    replace_collection: bool = False
    embedding: EmbeddingModelsConfig = OpenAIEmbeddingsConfig()
    use_jsonb: bool = True


class PGVector(VectorStore):
    def __init__(self, config: PGVectorConfig):
        super().__init__(config)
        self.config: PGVectorConfig = config
        self.embedding_fn = self.embedding_model.embedding_fn()
        self.embedding_dim = self.embedding_model.embedding_dims
        self._classes: Dict[str, Any] = {}

        if Vector is None:
            raise ImportError("pgvector is not installed")

        self.engine = create_engine(
            f"postgresql+psycopg2://{self.config.username}:{self.config.password}@"
            f"{self.config.host}:{self.config.port}/{self.config.database}",
            pool_size=10,
            max_overflow=20,
        )
        self._create_vector_extension(self.engine)

        self.SessionLocal = scoped_session(
            sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine,
            )
        )
        self.metadata = MetaData()
        self.metadata.reflect(bind=self.engine)
        self.EmbeddingTable = self._get_embedding_table_class(
            self.config.collection_name, self.embedding_dim
        )
        Base.metadata.create_all(bind=self.engine)

    def _create_vector_extension(self, conn: Engine) -> None:
        with conn.connect() as connection:
            with connection.begin():
                statement = text(
                    "SELECT pg_advisory_xact_lock(1573678846307946496);"
                    "CREATE EXTENSION IF NOT EXISTS vector;"
                )
                connection.execute(statement)

    def _get_embedding_table_class(
        self, table_name: str, vector_dimension: Optional[int] = None
    ) -> Any:
        class EmbeddingTable(Base):
            __tablename__ = table_name

            id = Column(
                String, nullable=False, primary_key=True, index=True, unique=True
            )
            embedding = Column(Vector(vector_dimension))
            document = Column(String, nullable=True)
            cmetadata = Column(JSONB, nullable=True)

            __table_args__ = (
                Index(
                    f"ix_{table_name}_cmetadata_gin",
                    "cmetadata",
                    postgresql_using="gin",
                    postgresql_ops={"cmetadata": "jsonb_path_ops"},
                ),
            )

        self._classes[table_name] = EmbeddingTable
        return EmbeddingTable

    def _parse_embedding_store_record(self, res: Any) -> Dict[str, Any]:
        metadata = res.cmetadata or {}
        window_ids_str = metadata.get("window_ids", "[]")
        try:
            window_ids = json.loads(window_ids_str)
        except json.JSONDecodeError:
            logger.warning(
                f"Could not decode window_ids for document {res.id}, "
                f"metadata: {metadata}. Using empty list."
            )
            window_ids = []

        if not isinstance(window_ids, list):
            logger.warning(
                f"window_ids for document {res.id} is not a list: {window_ids}."
                "Using empty list."
            )
            window_ids = []

        metadata["window_ids"] = window_ids
        metadata["id"] = res.id

        return {
            "content": res.document,
            "metadata": self.config.metadata_class(**metadata),
        }

    def clear_all_collections(self, really: bool = False, prefix: str = "") -> int:
        if not really:
            logger.warning("Not deleting all tables, set really=True to confirm")
            return 0

        with self.SessionLocal() as session:
            metadata = MetaData()
            metadata.reflect(bind=self.engine)
            deleted_count = 0
            for table_name in list(self._classes.keys()):
                if table_name.startswith(prefix):
                    self._classes.pop(table_name)

                    if table_name in metadata.tables:
                        table = metadata.tables[table_name]
                        table.drop(self.engine, checkfirst=True)

                    deleted_count += 1
            session.commit()
            logger.warning(f"Deleted {deleted_count} tables with prefix '{prefix}'.")
            return deleted_count

    def clear_empty_collections(self) -> int:
        with self.SessionLocal() as session:
            metadata = MetaData()
            metadata.reflect(bind=self.engine)
            deleted_count = 0
            for table_name in list(self._classes.keys()):
                EmbeddingTable = self._classes[table_name]

                embedding_count = session.query(EmbeddingTable).count()

                if embedding_count == 0:
                    if table_name in metadata.tables:
                        table = metadata.tables[table_name]
                        table.drop(self.engine, checkfirst=True)
                    self._classes.pop(table_name)
                    deleted_count += 1

            session.commit()
            logger.warning(f"Deleted {deleted_count} empty tables.")
            return deleted_count

    def list_collections(self, empty: bool = True) -> List[str]:
        with self.SessionLocal() as session:
            table_names = []
            for table_name in self._classes.keys():
                EmbeddingTable = self._classes[table_name]
                if empty:
                    table_names.append(table_name)
                else:
                    embedding_count = session.query(EmbeddingTable).count()
                    if embedding_count > 0:
                        table_names.append(table_name)
            return table_names

    def create_collection(self, collection_name: str, replace: bool = False) -> None:
        if replace:
            self.delete_collection(collection_name)

        self.set_collection(collection_name, replace=False)

    def set_collection(self, collection_name: str, replace: bool = False) -> None:
        if collection_name == self.config.collection_name:
            return

        super().set_collection(collection_name,  replace)

        self.EmbeddingTable = self._get_embedding_table_class(
            collection_name, self.embedding_dim
        )

        if not inspect(self.engine).has_table(collection_name):
            Base.metadata.create_all(bind=self.engine)

    @staticmethod
    def _generate_ids(documents: Sequence[Document]) -> List[str]:
        return [str(uuid.uuid5(uuid.NAMESPACE_DNS, d.content)) for d in documents]

    def add_documents(self, documents: Sequence[Document]) -> None:
        super().maybe_add_ids(documents)
        with self.SessionLocal() as session:
            embeddings = [self.embedding_fn([d.content])[0] for d in documents]
            ids = self._generate_ids(documents)
            metadatas = [d.metadata.dict() for d in documents]

            for m in metadatas:
                for k, v in m.items():
                    if isinstance(v, list):
                        m[k] = json.dumps(v)

            new_embeddings = [
                self.EmbeddingTable(
                    id=ids[i],
                    embedding=embeddings[i],
                    document=documents[i].content,
                    cmetadata=metadatas[i],
                )
                for i in range(len(documents))
            ]
            session.add_all(new_embeddings)
            session.commit()

    def similar_texts_with_scores(
        self, text: str, k: int = 5
    ) -> List[Tuple[Document, float]]:
        embedding = self.embedding_fn([text])[0]
        with self.SessionLocal() as session:
            results = (
                session.query(
                    self.EmbeddingTable,
                    self.EmbeddingTable.embedding.cosine_distance(embedding).label(
                        "distance"
                    ),
                )
                .order_by("distance")
                .limit(k)
                .all()
            )
            return [
                (
                    Document(
                        content=r.EmbeddingTable.content,
                        metadata=self.config.metadata_class(
                            **json.loads(r.EmbeddingTable.metadata)
                        ),
                    ),
                    r.distance,
                )
                for r in results
            ]

    def get_all_documents(self, where: str = "") -> List[Document]:
        with self.SessionLocal() as session:
            query = session.query(self.EmbeddingTable)
            results = query.all()

            documents = [
                Document(**self._parse_embedding_store_record(res)) for res in results
            ]
            return documents

    def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
        with self.SessionLocal() as session:
            query = session.query(self.EmbeddingTable).filter(
                self.EmbeddingTable.id.in_(ids),
            )
            results = query.all()

            documents = [
                Document(**self._parse_embedding_store_record(row)) for row in results
            ]
            return documents

    def delete_collection(self, collection_name: str) -> None:
        if collection_name in self._classes:
            self._classes.pop(collection_name)

            metadata = MetaData()
            metadata.reflect(bind=self.engine)
            table = metadata.tables[collection_name]
            table.drop(self.engine, checkfirst=True)

        else:
            logger.warning(f"Table '{collection_name}' not found.")
