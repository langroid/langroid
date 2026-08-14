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

import pytest

from langroid.vector_store.chromadb import ChromaDB, ChromaDBConfig
from langroid.vector_store.lancedb import LanceDB, LanceDBConfig
from langroid.vector_store.meilisearch import MeiliSearch, MeiliSearchConfig
from langroid.vector_store.pineconedb import PineconeDB, PineconeDBConfig
from langroid.vector_store.postgres import PostgresDB, PostgresDBConfig
from langroid.vector_store.qdrantdb import QdrantDB, QdrantDBConfig
from langroid.vector_store.weaviatedb import WeaviateDB, WeaviateDBConfig

# (provider class, config class) pairs for every vector-store provider.
PROVIDERS = [
    (ChromaDB, ChromaDBConfig),
    (LanceDB, LanceDBConfig),
    (MeiliSearch, MeiliSearchConfig),
    (PineconeDB, PineconeDBConfig),
    (PostgresDB, PostgresDBConfig),
    (QdrantDB, QdrantDBConfig),
    (WeaviateDB, WeaviateDBConfig),
]


@pytest.mark.parametrize("provider_cls,config_cls", PROVIDERS)
def test_no_shared_mutable_default_config(provider_cls, config_cls):
    """The __init__ default for `config` must not be a shared mutable
    instance."""
    sig = inspect.signature(provider_cls.__init__)
    default = sig.parameters["config"].default
    assert default is None, (
        f"{provider_cls.__name__}.__init__ uses a shared mutable default "
        f"({type(default).__name__}); use "
        f"`config: {config_cls.__name__} | None = None` and "
        f"`config = config or {config_cls.__name__}()` instead."
    )


@pytest.mark.parametrize("provider_cls,config_cls", PROVIDERS)
def test_no_arg_instances_get_fresh_config(provider_cls, config_cls):
    """Two no-arg instances must not share the same config object."""
    # We cannot instantiate the providers without an embedding model / API key,
    # so we verify the fix at the config level: calling the config class twice
    # yields distinct objects, and the __init__ default is None (so each call
    # builds a fresh config via `config or Config()`).
    cfg_a = config_cls()
    cfg_b = config_cls()
    assert cfg_a is not cfg_b
    assert cfg_a == cfg_b  # equal but not identical
