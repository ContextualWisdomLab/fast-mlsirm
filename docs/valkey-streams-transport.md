# Valkey Streams remote transport

`ValkeyStreamsBackend` publishes validated remote envelopes with `XADD` and
reads completed outcomes through a consumer group. `ValkeyStreamsOutcomeStore`
first reclaims idle pending records with `XAUTOCLAIM` (advancing until the
server returns cursor `0-0`), then reads new records with `XREADGROUP`,
validates the fingerprint against the serialized outcome, and acknowledges
each accepted record with `XACK`. Claim and fresh loops process bounded
`batch_size` fetches, persist/ACK after each batch, and re-check the remaining
wait deadline so a stream that keeps receiving messages cannot hang past the
deadline. Follow-up `XREADGROUP` calls omit `BLOCK` entirely — Redis/Valkey
treat `BLOCK 0` as wait-forever — and any blocking read is capped to the
remaining wait deadline. `wait_for_terminal` performs a final hash lookup
after the wait loop so a drain that persisted the last fingerprint at/after
the deadline does not report a false timeout. Durable lookup uses a companion
hash at ``{stream}:committed`` with ``HSETNX`` so the first successful outcome
matches the SQLite ledger contract across process restarts and consumer-group
members. ``run_batch`` drains until an explicit deadline instead of a single
batch, and publishes ``requested_device`` / ``effective_device`` on each job
envelope. Like ``SubprocessExecutor``, ``run_batch`` requires the numerical
``payload`` and rejects it before any ``XADD`` unless its canonical JSON
identity equals every envelope's ``manifest.payload_sha256``; the job record
carries it as a canonical-JSON ``payload`` field so workers never execute a
unit without its configuration. Outcomes return ordered by
``(unit_index, run_id)``.

Failed outcomes are terminal. The first failure per fingerprint is stored in
``{stream}:failed`` and acknowledged, so ``run_batch`` returns it beside the
successful units instead of timing out. A later success for the same
fingerprint still wins, and ``committed_success`` reports successes only.
Outcome records larger than 1 MiB (the worker stdout bound) or nested deeper
than 64 arrays/objects are rejected before they are acknowledged, and
``wait_timeout_s`` must be a finite positive number.

The adapter accepts a synchronous redis-py-compatible client supplied by the
host application; fast-mlsirm does not add a Valkey client dependency. The
consumer group must have permission for `XGROUP`, `XADD`, `XREADGROUP`,
`XAUTOCLAIM`, `XACK`, and hash `HSETNX`/`HGET`/`HGETALL`. Stream retention,
authentication, TLS, topology, worker deployment, and retry policy remain host
responsibilities.

## Worker

`fast_mlsirm.remote_worker.ValkeyStreamsWorker` is the reference consumer of
the job records above. Each `run_once()` pass reclaims idle pending jobs with
`XAUTOCLAIM`, so a crashed member's job is picked up by another member, then
reads new jobs with `XREADGROUP` and handles each one:

- a job whose fingerprint is already in the first-success hash is
  acknowledged without running again;
- the envelope, payload, and device fields are decoded under the same 1 MiB /
  depth-64 bounds, and the envelope must hash to the record's fingerprint;
- the worker checks its own manifest and the installed library version
  against the envelope cohort, then runs `execute_envelope`, which verifies
  payload identity and dispatches the family;
- it publishes a COMPLETED or FAILED outcome carrying the driver identity
  from the job record (`driver_host`, `driver_pid`) and its own provenance,
  then acknowledges the job.

A record that cannot be decoded into an envelope is acknowledged and counted
as poisoned so one bad record cannot stall the loop. The pass returns a
`ValkeyWorkerPass` with completed/failed/skipped/poisoned counts. Running
passes in a loop, process supervision, and retry policy stay with the host.

Protocol coverage lives in `tests/test_remote_exec_valkey.py` and
`tests/test_remote_exec_valkey_worker.py`;
`tests/test_remote_exec_family_equivalence.py` round-trips every remote family
through the backend and worker. Isolated
localhost daemon evidence is exercised by
`scripts/repro_valkey_defects.py` when `FAST_MLSIRM_VALKEY_URL` points at a
disposable instance; that script creates only UUID-suffixed keys and deletes
them on exit.
