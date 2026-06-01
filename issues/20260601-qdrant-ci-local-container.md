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

## Follow-up (not part of this change)

To actually stop the cloud bill, delete or suspend the Qdrant Cloud cluster
(and clear any retained backups) in the Qdrant Cloud console — deleting
collections alone does not stop the hourly cluster charge.
