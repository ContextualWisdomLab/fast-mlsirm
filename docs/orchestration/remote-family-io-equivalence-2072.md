# Remote family I/O, model, and version identity (#2072)

Status: acceptance evidence for PR #2072 follow-up. Does **not** close #2001,
implement Valkey transport, or claim A4 completion.

## Problem

`RemoteJobFamily` listed eight numerical families after the policy correction in
`9cb68d33`, but `fast_mlsirm.remote_worker` only dispatched three of them
(`mc_replicate`, `fit_restart`, `scoring_person`). Envelope admission therefore
looked family-complete while worker dispatch was not.

## Contract

Each remote unit carries three identity layers that must match before and after
dispatch:

| Layer | Field(s) | Fail-closed rule |
|-------|----------|------------------|
| Envelope | `envelope_fingerprint`, `input_identity_sha256` | Stable SHA-256 over canonical envelope JSON |
| Payload cohort | `manifest.payload_sha256`, `payload_ref` | Driver and worker both hash canonical payload JSON; mismatch raises `CohortMismatchError` before spawn |
| Cohort manifest | `schema_version`, `library_version`, `source_sha256`, `seed_derivation_rule`, `float_path`, optional `integration_nodes_sha256` | `SubprocessExecutor` rejects batches whose worker manifest is incompatible with any envelope manifest |
| Output | `output_identity_sha256`, `result` | Worker computes SHA-256 over canonical result JSON; driver stores it on `RemoteJobOutcome` |

The manifest comparison is a comparison of caller declarations, not installed
worker source, wheel, device or float-path attestation. Outcomes serialize
`identity_verification="declared_unverified"`; legacy records without that field
restore with the same unverified limitation. Worker version/host/PID/time
readback and result digests do not establish a measured source/device cohort.

Until a measured device-readback producer contract exists, subprocess dispatch
and the worker CLI accept only `cpu` requested/effective device declarations.
They reject other labels before executing the supplied numerical call. These
labels do not select or measure kernels: payload configuration can still select
its own backend/device, including an automatic path. Thus `cpu` here is an
allowed unverified declaration, not evidence of actual CPU execution. Installed
artifact/source, actual device and float-path acceptance remain outstanding.

Local↔remote equivalence for one family means: the in-process
`execute_envelope(...)` result and the `SubprocessExecutor` outcome for the same
envelope and payload share the same `output_identity_sha256`.

## Family → production API map

| `RemoteJobFamily` | Production entry point | Split policy |
|-------------------|------------------------|--------------|
| `mc_replicate` | `fast_mlsirm.simulate` | Independent replicate units |
| `fit_restart` | `fast_mlsirm.fit` | Independent restart seeds via `derive_index_seed` |
| `scoring_person` | `fast_mlsirm.score_wle` | Independent person shards |
| `em_m_step` | `fast_mlsirm.fit` with caller-owned `max_iter=1` | Whole call only |
| `se_derivatives` | `fast_mlsirm.bifactor_grm.bifactor_oakes_se` | Whole call only |
| `regression_contrasts` | `fast_mlsirm.regression.fit_ols_hc` + `contrast` | Whole call only |
| `fipc` | `fast_mlsirm.polytomous.fit_poly_fipc` | Whole call only |
| `two_tier` | `fast_mlsirm.two_tier_grm.fit_two_tier_grm` | Whole call only |

Internally unshardable families may still run on a remote host as one complete
call. `unit_index` is a nonnegative whole-call identifier and seed input; a
nonzero index is not evidence of internal partitioning. The worker adapters
execute the complete supplied call and do not interpret the index as an
internal shard selector.

`admit_remote_job_internal_shard` rejects an explicit internal-shard request.
Batch preflight also rejects more than one envelope with the same run/family
for an internally unshardable family, before any handler or worker dispatch.
This conservative batch guard does not enforce a run-wide invariant across separate batches.
Callers must not encode internal partitions as separate whole-call payloads or
use separate batches to bypass the explicit internal-shard policy. The ledger
keys envelopes for successful-result reuse; it is not a global run-policy
coordinator and does not prove that caller-supplied payloads are whole calls.

## Evidence

Focused pytest module:

```bash
python -m pytest tests/test_remote_exec_family_equivalence.py -q
python -m pytest tests/test_remote_exec.py -q
```

`tests/test_remote_exec_family_equivalence.py` asserts:

1. `RemoteJobFamily` enum members equal the handled worker set.
2. Every family executes through `execute_envelope` without `unsupported remote family`.
3. Local and subprocess paths agree on `output_identity_sha256`, input fingerprint,
   and `library_version` provenance.
4. A deliberate `library_version` cohort mismatch fails closed on dispatch.

## Explicit non-claims

- No Valkey/Redis Streams transport in this slice.
- No bootstrap adapter or cross-host always-on evidence.
- No formula or model-contract changes.
- Does not close #2001 or #2071 durable-store follow-ups.
