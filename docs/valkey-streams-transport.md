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
- envelope and payload JSON use the same 1 MiB / depth-64 bounds, the
  requested-device label is bounded, and the envelope must hash to the
  record's fingerprint;
- the worker checks its own manifest, payload identity, and the installed
  library version against the envelope cohort before invoking any executor,
  including an injected executor; the default dispatch runs the family;
- it publishes a COMPLETED or FAILED outcome carrying the driver identity
  from the job record (`driver_host`, `driver_pid`) and its own provenance,
  then acknowledges the job.

The worker preserves `requested_device` as request intent. Its result-only
executor interface does not attest the device actually used, so
`effective_device` is `unknown`, even if the job publisher supplied `cpu` or
`gpu`. This is not CPU/GPU execution evidence, and a requested GPU does not
prove that execution avoided a CPU fallback.

Execution is **at-least-once**, not exactly-once. An outcome `XADD` error
propagates without acknowledging the job, even when the append may have
succeeded but its reply was lost. An `XACK` error also propagates. A host retry
can therefore execute the same unit again before its success reaches the
first-success hash. Concurrent consumers of duplicate jobs can both execute;
`HSETNX` chooses one successful outcome, not one execution. Retry identity
uses the same envelope fingerprint and unit seed; deterministic executors
produce the same result digest. Injected executors must tolerate re-execution
and must not rely on this worker to protect external side effects.

Publication precedes job acknowledgement, but a successful `XADD` reply is
not a disk-durability guarantee. The host must configure persistence and
retention to meet its loss budget: RDB can lose writes since the last snapshot,
and AOF `everysec` can lose recent acknowledged writes. See the official
[acknowledgement contract](https://valkey.io/commands/xack/) and
[persistence guidance](https://valkey.io/topics/persistence/). The in-memory
fault tests below establish client ordering and result identity only; they do
not establish crash durability, installed-wheel provenance, multi-host
execution, or GPU attestation. Those remain separate acceptance evidence.

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
