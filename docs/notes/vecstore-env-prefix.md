# Vector-Store Config Env-Var Prefixes

!!! warning "Behavior change in 0.67.0"
    As of Langroid 0.67.0, **bare environment variables such as `HOST`,
    `PORT`, `COLLECTION_NAME`, and `FULL_EVAL` no longer configure
    vector stores.** If you relied on any of these (this was never
    documented, but it did work), you must rename them with the
    per-provider prefix from the table below, e.g.
    `QDRANT_HOST`, `QDRANT_PORT`, `VECDB_FULL_EVAL`.
    The documented variables (`QDRANT_API_KEY`, `QDRANT_API_URL`,
    `WEAVIATE_API_KEY`, etc.) are **unaffected**.

Vector-store configs are `pydantic-settings` classes, which means their
fields can be set via environment variables. Prior to this change (see
issue #1078), `VectorStoreConfig` and its subclasses declared **no
`env_prefix`**, so every field name was itself a case-insensitive
environment variable. A bare `HOST`, `PORT`, `CLOUD`, `TIMEOUT`,
`BATCH_SIZE`, `STORAGE_PATH`, `COLLECTION_NAME`, or `FULL_EVAL` in the
environment silently overrode the default for *every* vector-store
config:

```bash
HOST=evil.example.com PORT=9999 FULL_EVAL=true \
  python -c "from langroid.vector_store.qdrantdb import QdrantDBConfig; \
             c = QdrantDBConfig(); print(c.host, c.port, c.full_eval)"
# old behavior: evil.example.com 9999 True
```

This was dangerous for two reasons:

- `HOST` and `PORT` are routinely set by shells, containers, and CI
  systems, so vector-store clients could silently point at the wrong
  server.
- `FULL_EVAL=true` disabled the code-injection guard (see
  [code-injection-protection](code-injection-protection.md)) with no
  explicit opt-in anywhere in the user's code.

## New behavior

Each vector-store config now has its own env prefix, consistent with
the rest of the codebase (`OPENAI_`, `AZURE_OPENAI_`, `NEO4J_`, ...):

| Config class        | Env prefix     |
|---------------------|----------------|
| `VectorStoreConfig` | `VECDB_`       |
| `QdrantDBConfig`    | `QDRANT_`      |
| `ChromaDBConfig`    | `CHROMA_`      |
| `LanceDBConfig`     | `LANCEDB_`     |
| `PineconeDBConfig`  | `PINECONE_`    |
| `PostgresDBConfig`  | `POSTGRES_`    |
| `MeiliSearchConfig` | `MEILISEARCH_` |
| `WeaviateDBConfig`  | `WEAVIATE_`    |

Bare env vars (`HOST`, `FULL_EVAL`, ...) are now ignored by all of
these configs. To set a field via the environment, use the prefix of
the specific config class, e.g. `QDRANT_HOST=my.qdrant.example` or
`QDRANT_FULL_EVAL=true` for `QdrantDBConfig`. Every provider subclass
in langroid overrides the prefix with its own, so within langroid
`VECDB_*` vars affect only direct `VectorStoreConfig` instances.
However, pydantic inherits `model_config`, so a third-party subclass
of `VectorStoreConfig` that does not set its own `env_prefix` will
read `VECDB_*` vars.

A `port` value coming from the *environment* that matches the exact
Kubernetes service-link format `tcp://<host>:<digits>` (e.g.
`QDRANT_PORT=tcp://10.0.0.8:6333`, injected into every pod by a
service named `qdrant` when `enableServiceLinks` is on) is ignored
with a warning and the default port is used. This exception applies
only to the env settings source and only to that full format:
malformed `tcp://` junk in the environment, and any `tcp://...` value
passed explicitly to the constructor, fail validation as usual.

Explicit constructor arguments always take priority over environment
values (standard `pydantic-settings` precedence):

```python
config = QdrantDBConfig(host="10.0.0.5")  # wins over QDRANT_HOST
```

## Migration

If you (deliberately) relied on bare env vars to configure a
vector store, rename them with the appropriate prefix from the table
above, e.g. `COLLECTION_NAME=mydocs` becomes
`QDRANT_COLLECTION_NAME=mydocs`.

Env vars read via `os.getenv` outside the config classes are
unaffected, e.g. `QDRANT_API_KEY`, `PINECONE_API_KEY`,
`MEILISEARCH_API_KEY`, `MEILISEARCH_API_URL`, `WEAVIATE_API_KEY`, and
`WEAVIATE_API_URL` keep their existing meanings. Note that the
`QDRANT_*` and `PINECONE_*` prefixes overlap with those existing
names, but the config classes have no `api_key`/`api_url` fields, so
extra prefixed vars are ignored.
