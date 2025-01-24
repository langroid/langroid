import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from dotenv import load_dotenv

from sqlalchemy import (
    cast,
    create_engine,
    delete,
    func,
    select,
    text,
    inspect
)
from sqlalchemy.dialects.postgresql import JSON, JSONB, JSONPATH, UUID, insert
from sqlalchemy.engine import Engine, Inspector
from sqlalchemy.orm import (
    declarative_base,
    relationship,
    scoped_session,
    sessionmaker,
)
from sqlalchemy import text, Table, MetaData

from langroid.embedding_models.base import (
    EmbeddingModelsConfig,
)
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.mytypes import Document, DocMetaData
from langroid.utils.configuration import settings
from langroid.utils.output.printing import print_long_text
from langroid.vector_store.base import VectorStore, VectorStoreConfig
import os
import uuid
from sqlalchemy import Column, String, ForeignKey, Index
import uuid
from sqlalchemy.orm import declarative_base, relationship
from typing import (
    cast as typing_cast,
)
logger = logging.getLogger(__name__)
#------Add you embeddig model config-----------------
load_dotenv()
# embed_cfg = OpenAIEmbeddingsConfig(
#     model_type="openai",
#     dims=512,
#     api_key="studio_key",
#     api_base="http://localhost:5000"
# )
embed_cfg = OpenAIEmbeddingsConfig(
    model_type="openai",
)

#--------------------------------------------------------

try:
    from pgvector.sqlalchemy import Vector  # type: ignore
except ImportError:
    logger.warning(
        """
        pgvector not installed, so Postgres vector store will not work.
        To use Postgres, run:
        pip install pgvector
        """
    )
    Vector = None

Base = declarative_base()

_classes: Dict[str, Any] = {}

class PGVectorConfig(VectorStoreConfig):
    docker: bool = True
    collection_name: str | None = "temp"
    storage_path: str = ".postgres/data"
    host: str = "localhost"
    port: int = 5435
    username: str = "postgres"
    password: str = "postgres"
    database: str = "langroid"
    use_jsonb: bool = True

def _get_embedding_table_class(table_name: str, vector_dimension: Optional[int] = None) -> Any:
    if table_name in _classes:
        return _classes[table_name]

    class EmbeddingTable(Base):
        __tablename__ = table_name

        id = Column(String, nullable=False, primary_key=True, index=True, unique=True)
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

    _classes[table_name] = EmbeddingTable
    return EmbeddingTable

def _create_vector_extension(conn: Engine) -> None:
    with conn.connect() as connection:
        with connection.begin():
            statement = text(
                "SELECT pg_advisory_xact_lock(1573678846307946496);"
                "CREATE EXTENSION IF NOT EXISTS vector;"
            )
            connection.execute(statement)

class PGVector(VectorStore):
    def __init__(self, config: PGVectorConfig = PGVectorConfig()):
        super().__init__(config) # reintroducing this is failing table creation
        self.config: PGVectorConfig = config
        self.embedding_fn = self.embedding_model.embedding_fn() # embedding_model set in super().__init__
        self.embedding_dim = self.embedding_model.embedding_dims

        if Vector is None:
            raise ImportError("pgvector is not installed")

        self.engine = create_engine(
            f"postgresql+psycopg2://{self.config.username}:{self.config.password}@"
            f"{self.config.host}:{self.config.port}/{self.config.database}",
            pool_size=10,
            max_overflow=20,
        )
        _create_vector_extension(self.engine)

        self.SessionLocal = scoped_session(
            sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine,
            )
        )
        self.metadata = MetaData()
        self.metadata.reflect(bind=self.engine)
        self.EmbeddingTable = _get_embedding_table_class(
            self.config.collection_name, self.embedding_dim
        )
        Base.metadata.create_all(bind=self.engine)
    
    def refresh_metadata(self):
        self.metadata.clear()
        self.metadata.reflect(bind=self.engine)

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
        metadata['id'] = res.id

        return {
            "content": res.document,
            "metadata": self.config.metadata_class(**metadata),
        }

    def set_collection(self, collection_name: str, replace: bool = False) -> None:
        if collection_name in Base.metadata.tables:
            return
        
        super().set_collection(collection_name, replace)

        self.EmbeddingTable = _get_embedding_table_class(
            collection_name, self.embedding_dim
        )

        if not inspect(self.engine).has_table(collection_name):
            Base.metadata.create_all(bind=self.engine)

    def clear_all_collections(self, really: bool = False, prefix: str = "") -> int:
        if not really:
            logger.warning("Not deleting all tables, set really=True to confirm")
            return 0

        with self.SessionLocal() as session:
            metadata = MetaData()
            metadata.reflect(bind=self.engine)
            deleted_count = 0
            for table_name in list(_classes.keys()):
                if table_name.startswith(prefix):
                    EmbeddingTable = _classes.pop(table_name)
                    
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
            for table_name in list(_classes.keys()):
                EmbeddingTable = _classes[table_name]

                embedding_count = session.query(EmbeddingTable).count()

                if embedding_count == 0:
                    if table_name in metadata.tables:
                        table = metadata.tables[table_name]
                        table.drop(self.engine, checkfirst=True)
                    _classes.pop(table_name)
                    deleted_count += 1

            session.commit()
            logger.warning(f"Deleted {deleted_count} empty tables.")
            return deleted_count

    def list_collections(self, empty: bool = True) -> List[str]:
        with self.SessionLocal() as session:
            table_names = []
            for table_name in _classes.keys():
                EmbeddingTable = _classes[table_name]
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

    @staticmethod
    def _generate_ids(documents: Sequence[Document]) -> List[str]:
        return [str(uuid.uuid5(uuid.NAMESPACE_DNS, d.content)) for d in documents]

    def maybe_add_ids(self, documents: Sequence[Document]) -> None:
        for d in documents:
            if d.metadata.id is None or d.metadata.id == "":
                d.metadata.id = str(uuid.uuid4())

    def add_documents(self, documents: Sequence[Document]) -> None:
        self.maybe_add_ids(documents)
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

    def delete_collection(self, collection_name: str) -> None:
        if collection_name in _classes: # clear cached embedding table
            EmbeddingTable = _classes.pop(collection_name)
            self.metadata.clear() # Clear metadata cache
            metadata = MetaData()
            metadata.reflect(bind=self.engine)
            
            if collection_name in metadata.tables:
                table = metadata.tables[collection_name]
                table.drop(self.engine, checkfirst=True) # drop table and update db
           
        else:
            logger.warning(f"Table '{collection_name}' not found.")

    def get_all_documents(self, where: str = "") -> List[Document]:
        with self.SessionLocal() as session:
            query = session.query(self.EmbeddingTable)
            results = query.all()

            documents = [
                Document(**self._parse_embedding_store_record(res))
                for res in results
            ]
            return documents

    def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
        with self.SessionLocal() as session:
            query = session.query(self.EmbeddingTable).filter(
                self.EmbeddingTable.id.in_(ids),
            )
            results = query.all()

            documents = [
                Document(**self._parse_embedding_store_record(row))
                for row in results
            ]
            return documents

    def similar_texts_with_scores(
        self, text: str, k: int = 5
    ) -> List[Tuple[Document, float]]:
        pass

def test_create_collection():
    db_config = PGVectorConfig(
        collection_name="test_collection",
        username="postgres",
        embedding=embed_cfg,
    )
    try:
        postgres_db = PGVector(db_config)
    except Exception as e:
        print(f"Error initializing PGVector: {e}")
        return

    engine = postgres_db.engine

    postgres_db.create_collection("test_collection1", replace=True)

    postgres_db.refresh_metadata()
    inspector = inspect(engine)
    assert inspector.has_table("test_collection1"), "Table 'test_collection1' does not exist"
    
    postgres_db.delete_collection("test_collection1")

    postgres_db.refresh_metadata()
    inspector = inspect(engine)

    assert not inspector.has_table("test_collection1"), "Table 'test_collection1' was not deleted"

def test_list_collections():
    db_config = PGVectorConfig(
        username="postgres",
    )
    try:
        postgres_db = PGVector(db_config)
    except Exception as e:
        print(f"Error initializing PostgresDB: {e}")
        return

    postgres_db.create_collection("test_collection_1")
    postgres_db.create_collection("test_collection_2")
    postgres_db.create_collection("test_collection_3")

    collections = postgres_db.list_collections()
    print(collections)
    assert "test_collection_1" in collections, "Collection 'test_collection_1' not found"
    assert "test_collection_2" in collections, "Collection 'test_collection_2' not found"
    assert "test_collection_3" in collections, "Collection 'test_collection_3' not found"

    postgres_db.delete_collection("test_collection_1")
    postgres_db.delete_collection("test_collection_2")
    postgres_db.delete_collection("test_collection_3")

    collections = postgres_db.list_collections()
    print(collections)
    assert "test_collection_1" not in collections, "Collection 'test_collection_1' was not deleted"
    assert "test_collection_2" not in collections, "Collection 'test_collection_2' was not deleted"
    assert "test_collection_3" not in collections, "Collection 'test_collection_3' was not deleted"

def test_add_documents():
    db_config = PGVectorConfig(
        collection_name="test_add_docs_collection",
        username="postgres",
        embedding=embed_cfg
    )
    try:
        postgres_db = PGVector(db_config)
    except Exception as e:
        print(f"Error initializing PGVector: {e}")
        return

    postgres_db.create_collection("test_add_docs_collection")

    doc1 = Document(content="document 1", metadata={"id": "1", "info": "info1"})
    doc2 = Document(content="document 2", metadata={"id": "2", "info": "info2"})

    postgres_db.add_documents([doc1, doc2])

    docs = postgres_db.get_all_documents()
    print(docs)
    assert len(docs) == 2, "Incorrect number of documents added"
    assert any(doc.content == "document 1" for doc in docs), "Document 1 not found"
    assert any(doc.content == "document 2" for doc in docs), "Document 2 not found"

    postgres_db.delete_collection("test_add_docs_collection")

def test_get_documents_by_ids():
    db_config = PGVectorConfig(
        collection_name="test_get_docs_collection",
        username="postgres",
        embedding=embed_cfg
    )
    try:
        postgres_db = PGVector(db_config)
    except Exception as e:
        print(f"Error initializing PGVector: {e}")
        return

    postgres_db.create_collection("test_get_docs_collection")

    doc1 = Document(content="document 1", metadata={"id": "id-1"})
    doc2 = Document(content="document 2", metadata={"id": "id-2"})
    doc3 = Document(content="document 3", metadata={"id": "id-3"})

    postgres_db.add_documents([doc1, doc2, doc3])

    id_1 = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc1.content))
    id_3 = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc3.content))

    retrieved_docs = postgres_db.get_documents_by_ids([id_1, id_3])
    print(retrieved_docs)
    assert len(retrieved_docs) == 2, "Incorrect number of documents retrieved"
    assert any(doc.content == "document 1" for doc in retrieved_docs), "Document 1 not found"
    assert any(doc.content == "document 3" for doc in retrieved_docs), "Document 3 not found"

    postgres_db.delete_collection("test_get_docs_collection")

def test_get_all_documents():
    db_config = PGVectorConfig(
        collection_name="test_get_all_docs_collection",
        username="postgres",
        embedding=embed_cfg
    )
    try:
        postgres_db = PGVector(db_config)
    except Exception as e:
        print(f"Error initializing PGVector: {e}")
        return

    postgres_db.create_collection("test_get_all_docs_collection")

    doc1 = Document(content="document 1", metadata={"id": "1"})
    doc2 = Document(content="document 2", metadata={"id": "2"})
    doc3 = Document(content="document 3", metadata={"id": "3"})

    postgres_db.add_documents([doc1, doc2, doc3])

    all_docs = postgres_db.get_all_documents()
    print(all_docs)
    assert len(all_docs) == 3, "Incorrect number of documents retrieved"
    assert any(doc.content == "document 1" for doc in all_docs), "Document 1 not found"
    assert any(doc.content == "document 2" for doc in all_docs), "Document 2 not found"
    assert any(doc.content == "document 3" for doc in all_docs), "Document 3 not found"

    postgres_db.delete_collection("test_get_all_docs_collection")

def test_clear_all_collections():
    db_config = PGVectorConfig(
        username="postgres",
        embedding=embed_cfg
    )
    try:
        postgres_db = PGVector(db_config)
    except Exception as e:
        print(f"Error initializing PGVector: {e}")
        return

    postgres_db.create_collection("test_collection_1")
    postgres_db.create_collection("test_collection_2")
    postgres_db.create_collection("test_collection_3")

    doc1 = Document(content="doc1 in collection 1", metadata={"id": "d1"})
    doc2 = Document(content="doc2 in collection 2", metadata={"id": "d2"})
    doc3 = Document(content="doc3 in collection 3", metadata={"id": "d3"})
    postgres_db.add_documents([doc1])
    postgres_db.set_collection("test_collection_2")
    postgres_db.add_documents([doc2])
    postgres_db.set_collection("test_collection_3")
    postgres_db.add_documents([doc3])
    postgres_db.set_collection("test_collection_1")

    num_deleted = postgres_db.clear_all_collections(really=True)
    assert num_deleted == 3, "Incorrect number of collections deleted"

    collections = postgres_db.list_collections()
    assert len(collections) == 0, "Collections were not cleared"

# # Run the test functions
# test_create_collection()
# test_list_collections()
# test_add_documents()
# test_get_documents_by_ids()
# test_get_all_documents()
# test_clear_all_collections()