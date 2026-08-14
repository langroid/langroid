"""Regression tests for langroid issue #1078.

VectorStoreConfig extends pydantic_settings.BaseSettings but previously set
no env_prefix, so every field name (HOST, PORT, FULL_EVAL, ...) became a
case-insensitive bare env var. Unrelated bare env vars silently overrode the
defaults of every vector-store config — worst case, a bare FULL_EVAL=true
silently disabled code-injection sanitization.

Each provider config now sets its own env_prefix (QDRANT_, LANCEDB_, ...),
so only prefixed env vars configure a store and bare env vars no longer leak.
"""

from langroid.vector_store.lancedb import LanceDBConfig
from langroid.vector_store.qdrantdb import QdrantDBConfig


def test_prefixed_env_var_is_read(monkeypatch):
    """A prefixed env var (QDRANT_HOST) configures the matching provider."""
    monkeypatch.setenv("QDRANT_HOST", "qdrant.example.com")
    monkeypatch.setenv("QDRANT_FULL_EVAL", "true")
    cfg = QdrantDBConfig()
    assert cfg.host == "qdrant.example.com"
    assert cfg.full_eval is True


def test_bare_env_var_no_longer_hijacks(monkeypatch):
    """Bare HOST/PORT/FULL_EVAL must NOT override the config defaults."""
    monkeypatch.setenv("HOST", "evil.example.com")
    monkeypatch.setenv("PORT", "9999")
    monkeypatch.setenv("FULL_EVAL", "true")
    cfg = QdrantDBConfig()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 6333
    assert cfg.full_eval is False


def test_explicit_constructor_arg_wins_over_env(monkeypatch):
    """An explicit constructor arg still takes priority over the env var."""
    monkeypatch.setenv("QDRANT_HOST", "env.example.com")
    cfg = QdrantDBConfig(host="explicit.example.com")
    assert cfg.host == "explicit.example.com"


def test_provider_prefixes_are_isolated(monkeypatch):
    """A LANCEDB_ env var must not affect the QdrantDB config."""
    monkeypatch.setenv("LANCEDB_HOST", "lancedb.example.com")
    cfg = QdrantDBConfig()
    assert cfg.host == "127.0.0.1"
    # and the LanceDB config does read its own prefix
    ldb = LanceDBConfig()
    assert ldb.host == "lancedb.example.com"
