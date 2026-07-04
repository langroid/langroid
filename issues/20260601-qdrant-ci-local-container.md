# Qdrant CI tests: use local service container instead of Qdrant Cloud

## Problem

The `Pytest` CI workflow ran the vector-db test suite against **Qdrant Cloud**,
incurring an ongoing monthly cloud-cluster bill. The cost is driven by the
running cluster, not by the test collections themselves.

## Root cause

`.github/workflows/pytest.yml` set `QDRANT_API_KEY` / `QDRANT_API_URL` from
GitHub secrets that point at a real Qdrant Cloud cluster (secrets last updated
in 2023). Any test using `cloud=True` (the `QdrantDBConfig` default) or
`docker=True` reads those env vars and connects to that cloud URL.

A local `qdrant/qdrant` service container was added to the workflow in PR #926
(Oct 2025), but the secrets were never repointed at it — so the container sat
unused and the cloud-mode tests kept hitting Qdrant Cloud.

## Change

Override the two env vars directly in the workflow so they target the local
`qdrant` service container instead of the cloud secrets:

```yaml
QDRANT_API_KEY: local-dev-key
QDRANT_API_URL: http://localhost:6333
```

The cloud secrets are intentionally bypassed (left intact but now unreferenced
by any workflow).

## Effect

All Qdrant tests that run in CI now use local storage:

- `cloud=True` / `docker=True` tests connect to the local service container
  (`http://localhost:6333`).
- `cloud=False` tests continue to use embedded local storage.
- No test connects to Qdrant Cloud anymore.

This works with a single env change because no Qdrant test hardcodes a cloud
URL, and the only `QDRANT_API_URL` monkeypatch already points at localhost —
every cloud/server-mode test derives its target solely from `QDRANT_API_URL`.

## Verification

- Audited every `QdrantDBConfig` usage under `tests/` and classified its
  connection mode (embedded vs local-container vs cloud).
- Ran a routing proof plus an add / semantic-search / delete cycle against a
  local `qdrant/qdrant:v1.15.5` container, confirming `cloud=True` resolves to
  the local container.

## CI fixes uncovered by the migration

The first CI run against the empty, shared container surfaced two issues:

- **`punkt_tab` (pre-existing on `main`):** NLTK 3.9 renamed the `punkt`
  tokenizer resource to `punkt_tab`, but the workflow's `nltk.download` list
  only fetched `punkt`. `test_retriever_agent.py` builds an agent and calls
  `ingest()` at module import, so this raised `LookupError` at collection
  time. On `main` it was masked by the `continue-on-error` / `--lf` retry
  logic. Fixed by adding `punkt_tab` to the download list.
- **Collection-creation race:** `test_retriever_agent.py` created its Qdrant
  collection at module import with the default `cloud=True`. Under
  `pytest -n auto`, every xdist worker raced to create the same
  `test-retriever` collection on the shared container (409 Conflict, then an
  xdist "different tests collected" abort). Previously hidden because the
  cloud collection persisted across runs. Fixed by setting `cloud=False` (the
  `storage_path=":memory:"` already intended embedded, per-worker storage).
- **Shared-collection race under xdist (PR review P1):** the parametrized
  `qdrant_cloud` / `qdrant_hybrid_cloud` fixtures in `test_vector_stores.py`
  and the `qdrant_cloud` case in `test_doc_chat_agent.py` reused a fixed
  `test-<model>` collection, so parallel workers on the shared container could
  delete/recreate it mid-test (404/409/empty results). Fixed by suffixing
  those collection names with the xdist worker id (`PYTEST_XDIST_WORKER`).
  `test_doc_chat_relevance.py`'s `:memory:` config was switched to
  `cloud=False` (embedded) for the same reason.

## Follow-up (not part of this change)

To actually stop the cloud bill, delete or suspend the Qdrant Cloud cluster
(and clear any retained backups) in the Qdrant Cloud console — deleting
collections alone does not stop the hourly cluster charge.
