"""Regression tests for langroid issue #1079.

All vector-store providers previously used a mutable default argument for
their config (e.g. ``def __init__(self, config: QdrantDBConfig =
QdrantDBConfig())``).
Python evaluates that default once at import time, so every no-arg instance
received the SAME config object. Providers then mutated that shared config in
place (e.g. ``set_collection`` writes ``self.config.collection_name``), so state
set on one no-arg instance leaked into the next no-arg instance.

These tests assert that each provider's ``__init__`` no longer uses a mutable
default, and that two no-arg instances get distinct config objects.
"""

import inspect
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import CollectionStatus, Distance, PointStruct, VectorParams

import langroid.vector_store.lancedb as lancedb_module
import langroid.vector_store.meilisearch as meilisearch_module
import langroid.vector_store.pineconedb as pineconedb_module
import langroid.vector_store.postgres as postgres_module
import langroid.vector_store.qdrantdb as qdrantdb_module
import langroid.vector_store.weaviatedb as weaviatedb_module
from langroid.vector_store.base import VectorStore, VectorStoreConfig
from langroid.vector_store.chromadb import ChromaDB, ChromaDBConfig
from langroid.vector_store.lancedb import LanceDB, LanceDBConfig
from langroid.vector_store.meilisearch import MeiliSearch, MeiliSearchConfig
from langroid.vector_store.pineconedb import PineconeDB, PineconeDBConfig
from langroid.vector_store.postgres import PostgresDB, PostgresDBConfig
from langroid.vector_store.qdrantdb import QdrantDB, QdrantDBConfig
from langroid.vector_store.weaviatedb import WeaviateDB, WeaviateDBConfig

# (provider class, config class) pairs for every vector-store provider.
PROVIDERS: list[tuple[type[VectorStore], type[VectorStoreConfig]]] = [
    (ChromaDB, ChromaDBConfig),
    (LanceDB, LanceDBConfig),
    (MeiliSearch, MeiliSearchConfig),
    (PineconeDB, PineconeDBConfig),
    (PostgresDB, PostgresDBConfig),
    (QdrantDB, QdrantDBConfig),
    (WeaviateDB, WeaviateDBConfig),
]


@pytest.mark.parametrize("provider_cls,config_cls", PROVIDERS)
def test_no_shared_mutable_default_config(
    provider_cls: type[VectorStore], config_cls: type[VectorStoreConfig]
) -> None:
    """Verify each provider constructor has a safe ``config`` default."""
    sig = inspect.signature(provider_cls.__init__)
    default = sig.parameters["config"].default
    assert default is None, (
        f"{provider_cls.__name__}.__init__ uses a shared mutable default "
        f"({type(default).__name__}); use "
        f"`config: {config_cls.__name__} | None = None` and "
        f"`config if config is not None else {config_cls.__name__}()` instead."
    )


def _isolate_provider_io(
    provider_cls: type[VectorStore], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replace provider I/O boundaries with in-memory test doubles."""
    if provider_cls is ChromaDB:
        chromadb = ModuleType("chromadb")
        chromadb.Client = (  # type: ignore[attr-defined]
            lambda *_args, **_kwargs: MagicMock()
        )
        chromadb.config = SimpleNamespace(Settings=MagicMock)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "chromadb", chromadb)
    elif provider_cls is LanceDB:
        monkeypatch.setattr(lancedb_module, "load_dotenv", lambda: None)
        monkeypatch.setattr(lancedb_module, "has_lancedb", True)
        monkeypatch.setattr(
            lancedb_module,
            "lancedb",
            SimpleNamespace(connect=MagicMock()),
            raising=False,
        )
    elif provider_cls is MeiliSearch:
        monkeypatch.setattr(meilisearch_module, "load_dotenv", lambda: None)
        meilisearch = ModuleType("meilisearch_python_sdk")
        meilisearch.AsyncClient = MagicMock  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "meilisearch_python_sdk", meilisearch)
    elif provider_cls is PineconeDB:
        monkeypatch.setattr(pineconedb_module, "load_dotenv", lambda: None)
        monkeypatch.setattr(pineconedb_module, "has_pinecone", True)
        monkeypatch.setenv("PINECONE_API_KEY", "offline-test-key")
        client = MagicMock()
        client.list_indexes.return_value.names.return_value = []
        monkeypatch.setattr(pineconedb_module, "Pinecone", lambda **_: client)
        monkeypatch.setattr(PineconeDB, "embedding_dim", 1)
    elif provider_cls is PostgresDB:
        monkeypatch.setattr(postgres_module, "has_postgres", True)
        monkeypatch.setattr(postgres_module, "MetaData", MagicMock, raising=False)
        sqlalchemy = ModuleType("sqlalchemy")
        sqlalchemy_orm = ModuleType("sqlalchemy.orm")
        sqlalchemy_orm.sessionmaker = MagicMock  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "sqlalchemy", sqlalchemy)
        monkeypatch.setitem(sys.modules, "sqlalchemy.orm", sqlalchemy_orm)
        monkeypatch.setattr(PostgresDB, "_create_engine", MagicMock())
        monkeypatch.setattr(PostgresDB, "_create_vector_extension", MagicMock())
        monkeypatch.setattr(PostgresDB, "_setup_table", MagicMock())
    elif provider_cls is QdrantDB:
        monkeypatch.setattr(qdrantdb_module, "load_dotenv", lambda: None)
        client = MagicMock()
        client.collection_exists.return_value = False
        client.get_collection.return_value.status = CollectionStatus.GREEN
        client.get_collection.return_value.vectors_count = 0
        monkeypatch.setattr("qdrant_client.QdrantClient", lambda **_kwargs: client)
        monkeypatch.setattr(QdrantDB, "embedding_dim", 1)
    elif provider_cls is WeaviateDB:
        monkeypatch.setattr(weaviatedb_module, "load_dotenv", lambda: None)
        weaviate = ModuleType("weaviate")
        weaviate.connect_to_embedded = MagicMock()  # type: ignore[attr-defined]
        classes = ModuleType("weaviate.classes")
        init = ModuleType("weaviate.classes.init")
        init.Auth = MagicMock  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "weaviate", weaviate)
        monkeypatch.setitem(sys.modules, "weaviate.classes", classes)
        monkeypatch.setitem(sys.modules, "weaviate.classes.init", init)


def _clear_config_environment(
    config_cls: type[VectorStoreConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prevent ambient settings variables from changing constructor branches."""
    config_fields = {name.casefold() for name in config_cls.model_fields}
    for name in tuple(os.environ):
        if name.casefold() in config_fields:
            monkeypatch.delenv(name)


@pytest.mark.parametrize("provider_cls,config_cls", PROVIDERS)
def test_provider_config_identity_contract(
    provider_cls: type[VectorStore],
    config_cls: type[VectorStoreConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify real constructors create fresh configs and preserve supplied ones."""
    _clear_config_environment(config_cls, monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    _isolate_provider_io(provider_cls, monkeypatch)

    provider_a = provider_cls()
    provider_b = provider_cls()
    assert isinstance(provider_a.config, config_cls)
    assert isinstance(provider_b.config, config_cls)
    assert provider_a.config is not provider_b.config

    monkeypatch.setattr(config_cls, "__bool__", lambda _: False, raising=False)
    config = config_cls()
    assert not config

    provider = provider_cls(config)
    assert provider.config is config


@pytest.mark.parametrize("provider_cls,config_cls", PROVIDERS)
def test_provider_accepts_explicit_none_config(
    provider_cls: type[VectorStore],
    config_cls: type[VectorStoreConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify explicit ``config=None`` creates the provider's config type."""
    _clear_config_environment(config_cls, monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    _isolate_provider_io(provider_cls, monkeypatch)

    provider = provider_cls(config=None)

    assert isinstance(provider.config, config_cls)


def _seed_default_qdrant_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Prepare Qdrant's default path without making an embedding request."""
    _clear_config_environment(QdrantDBConfig, monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    monkeypatch.delenv("QDRANT_API_URL", raising=False)
    monkeypatch.setattr(qdrantdb_module, "load_dotenv", lambda: None)

    # load_dotenv() resolves the repo .env from the CALLING module's directory
    # (not cwd), so any langroid module doing its own load_dotenv() during
    # construction resurrects QDRANT_* vars deleted above. Neutralize at the
    # os.getenv level so this test always exercises local-storage mode.
    real_getenv = os.getenv

    def _getenv(name: str, default: str | None = None) -> str | None:
        if name.startswith("QDRANT_"):
            return default
        return real_getenv(name, default)

    monkeypatch.setattr(os, "getenv", _getenv)

    client = QdrantClient(path=".qdrant/data")
    try:
        client.create_collection(
            collection_name="temp",
            vectors_config=VectorParams(size=1, distance=Distance.COSINE),
        )
        client.upsert(
            collection_name="temp",
            points=[PointStruct(id=1, vector=[0.0])],
        )
    finally:
        client.close()


def test_no_arg_instances_get_fresh_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify real no-argument providers receive distinct configurations."""
    _seed_default_qdrant_store(tmp_path, monkeypatch)

    provider_a = QdrantDB()
    provider_a.close()
    provider_b = QdrantDB()

    try:
        assert provider_a.config is not provider_b.config
    finally:
        provider_b.close()
