"""Tests for env-var prefixes on vector-store configs (issue #1078).

`VectorStoreConfig` extends `BaseSettings`, so without an `env_prefix`
every field name was itself a case-insensitive environment variable:
a bare `HOST`, `PORT`, or worst `FULL_EVAL` in the environment silently
overrode the default for every vector-store config. These tests assert
that bare env vars no longer leak in, that properly prefixed env vars
(`VECDB_*` for the base config, `QDRANT_*` etc. per provider) do work,
and that explicit constructor arguments beat env values.

No live vector-store connections are needed: config construction only.
"""

import logging
import os
from typing import List, Type

import pytest
from pydantic import ValidationError
from pydantic_settings import BaseSettings

from langroid.utils.configuration import set_env
from langroid.vector_store.base import VectorStoreConfig
from langroid.vector_store.chromadb import ChromaDBConfig
from langroid.vector_store.lancedb import LanceDBConfig
from langroid.vector_store.meilisearch import MeiliSearchConfig
from langroid.vector_store.pineconedb import PineconeDBConfig
from langroid.vector_store.postgres import PostgresDBConfig
from langroid.vector_store.qdrantdb import QdrantDBConfig
from langroid.vector_store.weaviatedb import WeaviateDBConfig

ALL_CONFIGS: list[Type[VectorStoreConfig]] = [
    VectorStoreConfig,
    QdrantDBConfig,
    ChromaDBConfig,
    LanceDBConfig,
    PineconeDBConfig,
    PostgresDBConfig,
    MeiliSearchConfig,
    WeaviateDBConfig,
]

# provider config -> (its env prefix, a str field to probe with)
PREFIXED_CASES: list[tuple[Type[VectorStoreConfig], str, str]] = [
    (VectorStoreConfig, "VECDB_", "host"),
    (QdrantDBConfig, "QDRANT_", "host"),
    (ChromaDBConfig, "CHROMA_", "host"),
    (LanceDBConfig, "LANCEDB_", "storage_path"),
    (PineconeDBConfig, "PINECONE_", "metric"),
    (PostgresDBConfig, "POSTGRES_", "host"),
    (MeiliSearchConfig, "MEILISEARCH_", "primary_key"),
    (WeaviateDBConfig, "WEAVIATE_", "host"),
]


# every field name these tests assert on, in env-var (upper) form
FIELD_VARS: list[str] = [
    "HOST",
    "PORT",
    "CLOUD",
    "TIMEOUT",
    "BATCH_SIZE",
    "TYPE",
    "STORAGE_PATH",
    "COLLECTION_NAME",
    "FULL_EVAL",
    "METRIC",
    "PRIMARY_KEY",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove ambient bare and prefixed env vars before each test.

    A runner may legitimately have e.g. `POSTGRES_HOST` set (docker/CI)
    or a stray `QDRANT_FULL_EVAL`; without this cleanup those would make
    the default-value assertions below fail spuriously. Runs before each
    test body, so tests that intentionally `setenv` still work.
    """
    prefixes = [""] + [prefix for _, prefix, _ in PREFIXED_CASES]
    for prefix in prefixes:
        for field_var in FIELD_VARS:
            monkeypatch.delenv(f"{prefix}{field_var}", raising=False)


@pytest.mark.parametrize("config_cls", ALL_CONFIGS)
def test_bare_env_vars_do_not_leak(
    config_cls: Type[VectorStoreConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bare (unprefixed) env vars must not override config defaults."""
    monkeypatch.setenv("HOST", "evil.example.com")
    monkeypatch.setenv("PORT", "9999")
    monkeypatch.setenv("FULL_EVAL", "true")
    monkeypatch.setenv("COLLECTION_NAME", "hijacked")
    monkeypatch.setenv("CLOUD", "true")
    monkeypatch.setenv("STORAGE_PATH", "/evil/path")

    defaults = config_cls.model_fields
    config = config_cls()
    assert config.host == defaults["host"].default
    assert config.port == defaults["port"].default
    assert config.full_eval is False
    assert config.collection_name == defaults["collection_name"].default
    assert config.cloud == defaults["cloud"].default
    assert config.storage_path == defaults["storage_path"].default


@pytest.mark.parametrize("config_cls,prefix,field", PREFIXED_CASES)
def test_prefixed_env_vars_are_honored(
    config_cls: Type[VectorStoreConfig],
    prefix: str,
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env vars with the class's own prefix DO override defaults."""
    monkeypatch.setenv(f"{prefix}{field.upper()}", "from-env")
    monkeypatch.setenv(f"{prefix}FULL_EVAL", "true")

    config = config_cls()
    assert getattr(config, field) == "from-env"
    assert config.full_eval is True


@pytest.mark.parametrize("config_cls,prefix,field", PREFIXED_CASES)
def test_constructor_args_beat_env_vars(
    config_cls: Type[VectorStoreConfig],
    prefix: str,
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit constructor kwargs take priority over prefixed env vars."""
    monkeypatch.setenv(f"{prefix}{field.upper()}", "from-env")
    monkeypatch.setenv(f"{prefix}FULL_EVAL", "true")

    config = config_cls(**{field: "explicit", "full_eval": False})
    assert getattr(config, field) == "explicit"
    assert config.full_eval is False


@pytest.mark.parametrize("config_cls,prefix,field", PREFIXED_CASES)
def test_other_providers_prefix_does_not_leak(
    config_cls: Type[VectorStoreConfig],
    prefix: str,
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One provider's prefixed env vars must not affect other providers."""
    monkeypatch.setenv(f"{prefix}FULL_EVAL", "true")
    for other_cls, other_prefix, _ in PREFIXED_CASES:
        if other_prefix == prefix:
            continue
        assert (
            other_cls().full_eval is False
        ), f"{prefix}FULL_EVAL leaked into {other_cls.__name__}"


@pytest.mark.parametrize(
    "config_cls,env_var",
    [
        (QdrantDBConfig, "QDRANT_PORT"),
        (PostgresDBConfig, "POSTGRES_PORT"),
    ],
)
def test_service_link_port_is_ignored_with_warning(
    config_cls: Type[VectorStoreConfig],
    env_var: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A k8s/docker service-link `tcp://...` port is ignored with a warning.

    With `enableServiceLinks` (on by default in Kubernetes), a service
    named e.g. `qdrant` injects `QDRANT_PORT=tcp://10.0.0.8:6333` into
    every pod; construction must succeed with the default port.

    Warnings are captured with a plain handler (not caplog) because CI
    disables pytest's logging plugin (`PYTEST_ADDOPTS="-p no:logging"`).
    """
    monkeypatch.setenv(env_var, "tcp://10.0.0.8:6333")
    default_port = config_cls.model_fields["port"].default

    records: List[logging.LogRecord] = []

    class Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("langroid.vector_store.base")
    handler = Collector(level=logging.WARNING)
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        config = config_cls()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    assert config.port == default_port
    text = "\n".join(r.getMessage() for r in records)
    assert "tcp://10.0.0.8:6333" in text
    assert "ignor" in text.lower()


def test_non_service_link_bad_port_still_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the `tcp://` service-link format is forgiven; other junk fails."""
    monkeypatch.setenv("QDRANT_PORT", "not-a-port")
    with pytest.raises(ValidationError):
        QdrantDBConfig()


def test_malformed_tcp_env_port_still_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `tcp://` value NOT matching the full service-link shape fails.

    Only `tcp://<host>:<digits>` (the exact k8s injection format) is
    forgiven; malformed `tcp://` junk must fail validation loudly.
    """
    monkeypatch.setenv("QDRANT_PORT", "tcp://garbage")
    with pytest.raises(ValidationError):
        QdrantDBConfig()


def test_constructor_tcp_port_still_fails() -> None:
    """The service-link exception applies to the env source ONLY.

    An explicit constructor value is a programming error, not a k8s
    artifact, so it must fail validation rather than be silently
    replaced with the default.
    """
    with pytest.raises(ValidationError):
        QdrantDBConfig(port="tcp://10.0.0.8:6333")  # type: ignore[arg-type]


def test_valid_env_port_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    """A normal numeric prefixed port env var is honored as before."""
    monkeypatch.setenv("QDRANT_PORT", "7777")
    assert QdrantDBConfig().port == 7777


def test_valid_constructor_port_still_works() -> None:
    """A normal explicit constructor port is honored as before."""
    assert QdrantDBConfig(port=7777).port == 7777


def test_trailing_newline_tcp_port_still_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A service-link-like value with a trailing newline fails validation.

    The full-format check must anchor at the true end of the string
    (`\\Z`), not `$` (which matches before a trailing newline), so this
    value is NOT silently discarded.
    """
    monkeypatch.setenv("QDRANT_PORT", "tcp://10.0.0.8:6333\n")
    with pytest.raises(ValidationError):
        QdrantDBConfig()


def test_set_env_writes_prefixed_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`set_env` writes names a fresh config actually reads back.

    `set_env` used to write bare upper-cased field names (`HOST`),
    which prefixed configs no longer read; it must write
    `<PREFIX><FIELD>` (e.g. `QDRANT_HOST`).
    """
    config = QdrantDBConfig(host="example.internal")
    # pre-register every var set_env will write, so monkeypatch teardown
    # restores each to its original (absent) state after the test
    for field_name, field in QdrantDBConfig.model_fields.items():
        name = field.alias or f"QDRANT_{field_name.upper()}"
        monkeypatch.setenv(name, "placeholder")
    set_env(config)
    assert os.environ["QDRANT_HOST"] == "example.internal"
    # pre-existing set_env limitation, unrelated to prefixes: complex
    # fields (nested configs, classes) are written as non-JSON python
    # reprs the env source cannot parse back; clear them so the fresh
    # construction below exercises the simple fields
    for name in (
        "QDRANT_EMBEDDING",
        "QDRANT_EMBEDDING_MODEL",
        "QDRANT_DOCUMENT_CLASS",
        "QDRANT_METADATA_CLASS",
    ):
        monkeypatch.delenv(name)
    assert QdrantDBConfig().host == "example.internal"


def test_set_env_bare_names_for_prefixless_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A settings class without env_prefix still gets bare var names."""

    class PlainSettings(BaseSettings):
        some_field: str = "default"

    monkeypatch.setenv("SOME_FIELD", "placeholder")
    set_env(PlainSettings(some_field="written"))
    assert os.environ["SOME_FIELD"] == "written"
    assert PlainSettings().some_field == "written"
