# Valkey Streams remote transport

`ValkeyStreamsBackend` publishes validated remote envelopes with `XADD` and
reads completed outcomes through a consumer group. `ValkeyStreamsOutcomeStore`
first reclaims idle pending records with `XAUTOCLAIM` (advancing until the
server returns cursor `0-0`), then reads new records with `XREADGROUP`,
validates the fingerprint against the serialized outcome, and acknowledges
each accepted record with `XACK`. Follow-up `XREADGROUP` calls omit `BLOCK`
entirely — Redis/Valkey treat `BLOCK 0` as wait-forever — and any blocking
read is capped to the remaining wait deadline. Durable lookup uses a
companion hash at ``{stream}:committed`` with ``HSETNX`` so the first
successful outcome matches the SQLite ledger contract across process restarts
and consumer-group members. ``run_batch`` drains until an explicit deadline
instead of a single batch, and publishes ``requested_device`` /
``effective_device`` on each job envelope.

The adapter accepts a synchronous redis-py-compatible client supplied by the
host application; fast-mlsirm does not add a Valkey client dependency. The
consumer group must have permission for `XGROUP`, `XADD`, `XREADGROUP`,
`XAUTOCLAIM`, `XACK`, and hash `HSETNX`/`HGET`/`HGETALL`. Stream retention,
authentication, TLS, topology, worker deployment, and retry policy remain host
responsibilities.

Protocol coverage lives in `tests/test_remote_exec_valkey.py`. Isolated
localhost daemon evidence is exercised by
`scripts/repro_valkey_defects.py` when `FAST_MLSIRM_VALKEY_URL` points at a
disposable instance; that script creates only UUID-suffixed keys and deletes
them on exit.
