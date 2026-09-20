# Valkey Streams remote transport

`ValkeyStreamsBackend` publishes validated remote envelopes with `XADD` and
reads completed outcomes through a consumer group. `ValkeyStreamsOutcomeStore`
first reclaims idle pending records with `XAUTOCLAIM`, then reads new records
with `XREADGROUP`, validates the fingerprint against the serialized outcome,
and acknowledges each accepted record with `XACK`. Durable lookup uses a
companion hash at ``{stream}:committed`` with ``HSETNX`` so the first
successful outcome matches the SQLite ledger contract across process restarts
and consumer-group members. ``run_batch`` drains until an explicit deadline
instead of a single batch.

The adapter accepts a synchronous redis-py-compatible client supplied by the
host application; fast-mlsirm does not add a Valkey client dependency. The
consumer group must have permission for `XGROUP`, `XADD`, `XREADGROUP`,
`XAUTOCLAIM`, and `XACK`. Stream retention, authentication, TLS, topology,
worker deployment, and retry policy remain host responsibilities.

This checkout had neither a Valkey/Redis daemon nor a Python client installed,
so verification used the deterministic protocol harness in
`tests/test_remote_exec_valkey.py`. That proves command flow, reclaim,
acknowledgement, and duplicate resolution, but is not production-host transport
evidence. A deployment must rerun the same contract against its real client and
server before claiming transport readiness.
