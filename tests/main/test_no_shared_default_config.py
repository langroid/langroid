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
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

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


def _seed_default_qdrant_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Prepare Qdrant's default path without making an embedding request."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    monkeypatch.setenv("CLOUD", "false")

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


class FalsyQdrantDBConfig(QdrantDBConfig):
    """Valid Qdrant configuration that behaves as a falsy value."""

    def __bool__(self) -> bool:
        """Return ``False`` to exercise identity-safe constructor handling."""
        return False


def test_falsy_config_is_preserved_by_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a supplied falsy configuration is not silently replaced."""
    _seed_default_qdrant_store(tmp_path, monkeypatch)
    config = FalsyQdrantDBConfig(
        cloud=False,
        collection_name=None,
        storage_path=":memory:",
    )

    provider = QdrantDB(config)

    try:
        assert provider.config is config
    finally:
        provider.close()
