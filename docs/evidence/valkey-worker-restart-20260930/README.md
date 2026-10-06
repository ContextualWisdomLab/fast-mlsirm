# Valkey worker graceful-restart evidence

Date: 2026-09-30 (KST). Tested worker implementation:
`3abb2596de1bdccf32ddf4be01e0659b2cfdc932` (#2283, draft).

## Scope and limits

Production MC dispatch ran in the existing editable installation on the Mac.
A disposable Valkey 8.1.10 instance ran on S1 behind an owned localhost SSH
tunnel. The manifest hashes came from test fixtures. This is **graceful daemon
restart recovery**, not installed-wheel/source attestation, multi-host numerical
execution, GPU attestation, exactly-once execution, SIGKILL crash recovery, or
host/power-loss durability.

The infrastructure owner performed `docker restart --time5` (graceful SIGTERM)
on the exact disposable container with the same private 32 MiB host tmpfs AOF
bind. The probe checked `appendonly=yes` and `appendfsync=always`. This tmpfs
survives the tested daemon restart, not a host or power failure.

Initial setup did **not** pass: `docker run -d` started the container, but a
subsequent `docker port` lookup found no public mapping and returned exit 1.
The owner recovered access to the same running container with a localhost-only
SSH tunnel into its private network. No second daemon was created and network
isolation was not relaxed. This is the observed failure and recovery; it does
not establish a Docker-engine root cause. The setup summary retains that
nonzero status and identifies the original receipt by digest. Supplemental
image metadata is a peer transcription of earlier preflight tool outputs, not
a new or independent inspection.

## Observed results

| Event | Actual KST timestamp | Result |
| --- | --- | --- |
| Instance start (infrastructure receipt) | 12:33:33.106 | Cached image; private instance |
| Before phase | 12:36:11.496–12:36:12.499 | Exit 0; 1.003 seconds |
| Graceful restart (infrastructure receipt) | 12:37:11.820–12:37:14.665 | Same container and AOF mount |
| After phase | 12:38:22.264–12:38:26.587 | Exit 0; 4.320 seconds |
| Infrastructure cleanup (owner receipt) | 12:39:46.154 | Complete before 12:41:35 deadline |

Before restart, two actual jobs were pending, one in each job stream. One job
had published a completed outcome and its first-success hash was committed,
but its ACK was interrupted. The other job's publication was interrupted
before append; its outcome stream was empty.

After restart, the server `run_id` changed and both jobs were still pending.
Each pending count went from 1 to 0 after recovery. The already committed job
was acknowledged without another executor invocation; the unpublished job
executed once. Envelope fingerprint, input identity, deterministic unit seed,
output digest and recomputed result digest matched. Effective device remained
`unknown` despite a producer-supplied GPU label. First-success count was one
per fingerprint. `identities.json` binds these checks to the test envelopes.

Each phase had an outer 60-second subprocess deadline and 3-second socket /
connect timeouts, with BLAS/OpenMP threads limited to one. Optimized Python is
rejected before third-party imports so evidence checks cannot be disabled.
The after phase deleted only its UUID-owned streams and companion hashes.

## Reproduction

Use a **new disposable** localhost endpoint with AOF enabled and
`appendfsync=always`; never restart a shared service. The scripts do not start,
restart, or remove a server. The infrastructure owner must bound its lifetime
and remove its exact owned container, network, tunnel and data mount even if a
probe fails. This example requires redis-py already installed in the project
environment; it adds no package runtime dependency.

From the supported checkout with its built extension:

```bash
export PYTHONPATH="$PWD/python:$PWD/tests"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
unset PYTHONOPTIMIZE
# Choose a new state path and receipt paths; existing files are not overwritten.
.venv/bin/python scripts/repro_valkey_worker_restart.py before \
  --port "$OWNED_LOCAL_PORT" --state "$NEW_STATE_FILE" --evidence "$BEFORE_RECEIPT"
# Owner restarts only the disposable server, then confirms its endpoint is ready.
.venv/bin/python scripts/repro_valkey_worker_restart.py after \
  --port "$OWNED_LOCAL_PORT" --state "$NEW_STATE_FILE" --evidence "$AFTER_RECEIPT"
# Owner removes the disposable instance and checks that its resources are absent.
```

The stored probe preserves the tested phase logic byte-for-byte. Its launcher
uses a checkout-relative probe path instead of the original temporary path.
Test fixtures are intentionally used: this does not certify source or installed
wheel provenance. The original phase receipts below are retained without
rewriting their timestamps or outputs.

## Receipt identities

- `before.json` SHA-256:
  `eadd37c4921a8d25099739e3893e1f875b9e86ec9bdde26260995ede2004e55a`
- `after.json` SHA-256:
  `27cfa7601e2c4679532da8b45e330d9c0c6166d6177f029b9f091a2e43ac2af1`
- Infrastructure receipt references and script hashes are recorded in
  `receipt-index.json`; they distinguish direct phase observations from
  infrastructure-owner observations.

Infrastructure limits: 128 MiB including swap, 0.5 CPU, 64 PIDs, maxmemory
8 MiB/noeviction, isolated network and no public host listener. Fresh preflight
at 12:32:44 KST reported 20.4 GiB available and load 5.34. No image was pulled.
The owner reported container/network absent, tmpfs unmounted and directory
absent, tunnel PID absent, and watchdog cancelled after cleanup. Local endpoint
closure was independently checked after that report.

## Other verification and remaining gates

The worker revision passed the targeted remote suite: **135 passed** in the
final pre-push run (47.57 seconds). An isolated filesystem overlay with #2286
implementation `5d39cfc3fc3eaa56d9051252cab4adfbb701d434` passed **113 tests** in
11.49 seconds. The docs-only #2286 tip
`0e4e149d508b8416299e9e9335e52cd28a3a0f60` was not the executed overlay target.
No Rust rebuild or full pytest run was performed.

After preserving the evidence, the same targeted remote suite passed again:
**135 passed in 185.38 seconds**. At 12:56 KST the host load was 64.51 and swap
headroom approximately 942 MiB. A separate normal-probe `--help` cold-import
check exceeded its 15-second auxiliary deadline; it was not described as a
pass or repeatedly retried. Lightweight receipt-hash checks, the launcher entry
point, and the early optimized-mode rejection all passed. The actual before /
after daemon phases had already passed their independent 60-second bounds.

#2283 remains draft. Worker-owned source/quadrature/device attestation,
installed-wheel and multi-host numerical evidence, and CPU/GPU evidence remain
independent gates. No result here proves exactly-once computation.
