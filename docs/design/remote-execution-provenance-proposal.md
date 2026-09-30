---
status: proposed
date: 2026-09-30
decision-makers: [L4 lead, numerical-core owner, release-evidence owner, bootstrap-consumer owner]
consulted: [L4 coordinator, Valkey worker owner]
---

# Worker-observed execution provenance

Working MADR-shaped proposal for joint review, not an accepted ADR, a finalized
wire schema, or implementation approval. No runtime or formula change is made
by this document. Architecturally significant drivers: security, maintainability,
and evolvability. Related: #2001, #2072, #2071, #2073, #2283, #2286.

## Context and Problem Statement

The current remote manifest describes a requested cohort. Comparing two copies
of its source, integration and precision declarations does not observe what ran.
The current worker checks payload identity and installed version, but returns
`unknown` for device because its executor interface supplies no execution-device
observation. The graceful-restart evidence is explicitly not source, wheel,
quadrature or device verification.

Read-only anchors in the tested checkout:

- `python/fast_mlsirm/remote_exec.py:189`: manifest fields;
  `:251`: cohort equality compares declarations.
- `python/fast_mlsirm/remote_worker.py:81`: fit-result serialization contains
  numeric/result identity but no observed device; `:376`: worker processing.
- `crates/mlsirm-core/src/lib.rs:198` and `:235`: GPU success and CPU fallback
  return the same numerical tuple without an execution-path receipt.
- `scripts/ci/release_artifact_transport.py:92`: release inventory binds a
  40-hex source commit separately from archive/member SHA-256 identities;
  `:263`: runtime verification binds a selected target/wheel, imported extension,
  source lock and accounted native members.
- `scripts/ci/capture_release_build_scope.py:27`: installed-file inventory
  hashes actual regular files and rejects unsafe/unaccounted members.

These existing build/runtime checks are reuse points, not proof that the
running worker has already performed them. Receipt existence alone is not
verification. Reuse their identity, target, imported-member and accounted-file
verification semantics, not the complete release-job receipt or its operational
assumptions. The `uv 0.12.5` pin and `build_env` fields in release validation are
CI/release context, not universal live-worker requirements.

Keep CI scripts as tooling. Define the smallest package-safe verification
boundary for independently approved artifact policy and actual local files;
this must not add a runtime dependency on CI scripts, build tools or a release
job's complete environment. The independently trusted policy must bind the
approved bundle digest and per-target members. A bundle or file inventory sent
by the producer is data to validate, never its own approval authority. Even
verified artifact readiness does not establish the nodes or device used.

## Decision Drivers

1. Separate producer requests from observations made by the trusted worker and
   numerical owner. Missing evidence must not be filled from request metadata.
2. Preserve fail-closed whole-batch admission, seed derivation, result identity,
   missing/rejected-replicate semantics and Rust/reference parity.
3. Accept independently approved artifacts for each target of one release;
   do not demand identical Linux and macOS wheel hashes.
4. Keep replay admission sensitive to evidence policy and schema. A legacy
   cache hit must not satisfy a stricter execution-evidence request.

## Considered Options

- **Echo the current manifest/requested device.** Smallest diff, but cannot
  detect source substitution, different weights or a GPU-to-CPU fallback.
- **Transport-generated evidence from config or adapter availability.** Can
  show availability and intent; still cannot show the actual numerical path.
- **Versioned worker/numerical-owner observations, bound to results.** Requires
  coordinated contracts and instrumentation but addresses the observed gaps.

## Decision Outcome (Proposed)

Prefer the third option, reusing existing envelope/result identities and
validated release/runtime evidence before adding abstractions. Treat the
result as **trusted-worker execution provenance**, not cryptographic/hardware
remote attestation against a malicious worker. Authentication and trusted
artifact-policy distribution belong to the host; this library must not add a
hosted assessment runtime or invent a signature scheme.

### Pre-execution readiness

The worker independently verifies its selected target and installed Python /
numerical-extension files against an approved release policy not supplied by
the job being verified. Bind imported code to those actual verified files,
including the loaded extension; a version string, filename or self-reported
RECORD is insufficient. A 40-hex source commit, a source/archive SHA-256, a wheel
archive SHA-256 and installed member hashes are different identities. Installed
file hashing does not reconstruct a wheel archive digest.

A post-import path hash or RECORD lookup cannot prove the bytes already loaded
into a process. The strict profile therefore needs a host-controlled approved
immutable installation and a startup/admission observation bound to that
worker process and job. Document the exact trust and immutability assumptions.
Any installation change invalidates the readiness/cached artifact observation;
restart/re-admit against the approved artifacts rather than claiming that a
later hash detected arbitrary loaded-code substitution. If those assumptions
cannot be established, evidence stays unverified. This is a provenance / TOCTOU
limit, not protection against a malicious worker or privileged host.

Recompute payload identity and derived seed from received content. Negotiate
whether the worker can observe required integration and execution-path facts
before starting work. Refuse incompatible readiness without running the fit.
Declared capabilities alone cannot replace post-execution observations.

### Post-execution evidence

The numerical owner reports the actual nodes **and weights** used, dimensions,
shape, ordering, dtype/endianness and node-generation contract. Counts and
producer-supplied hashes do not establish this identity. A family that does not
use quadrature needs an explicit applicability rule, not a fabricated hash.
Adaptive/multi-stage integration needs a defined observation scope; do not
silently call the last or planned grid the grid used by the whole execution.

Observe effective device, computation precision and actual fallback or mixed
paths at the numerical decision point. Do not infer these from requested
config, adapter availability or a cast of the final output array. A GPU request
that ran on CPU is CPU execution; acceptance depends on an explicit policy.
Scheduling/thread allocation and numerical chunking remain distinct.

Bind observations to envelope fingerprint, run/unit/family, payload identity,
derived seed and the recomputed output digest. Consumer admission checks every
row and the complete batch before aggregation, including cached/reclaimed
outcomes. A result digest binds returned content; it is not proof of numerical
parity across architectures or precision paths.

Pre-execution readiness and post-execution checks are separate. A mismatch
invalidates the unit/cohort without statistical resampling. It must never be
encoded as `COMPLETED + rejected`: documented statistical rejection and an
execution-provenance failure are different contracts.

A documented statistical rejection can also occur before numerical execution.
After validated admission, retain that legitimate rejection and report explicit
not-executed / non-applicable observations bound to its input, logical unit,
policy and rejection output. Do not invent nodes, device or precision for a
kernel that never ran; missing required evidence and approved non-applicability
are distinct. Replaying such a rejection still requires the current strict
policy and cannot inherit trust merely from its archived statistical flag.

### Compatibility and replay

Use a coordinated versioned execution-evidence contract and capability
negotiation; exact version/serialization remains unresolved. Do not reinterpret
schema-1 records or silently attach evidence copied from their manifests.
Legacy records can remain readable as explicitly unverified but cannot satisfy
strict #2001 acceptance. Missing/unknown required observations fail strict
admission; they do not become CPU, GPU or f64 by default.

Evidence-schema and policy identities participate in canonical replay admission,
so old successful hashes cannot bypass stricter validation. Keep logical unit
identity and seed derivation separate from policy-aware admission/cache
identity: the same base seed and unit index retain the same derived seed and
resample when evidence policy changes. Do not feed policy/schema/cache digests
into the RNG seed or automatically redraw a refused replay. Re-execution, if
explicitly permitted, uses that original logical unit and seed. Retain archived
schema-1 results as unverified; do not delete them or rewrite their numerical
identities during migration. Finalize the fingerprint/cache transition before
writing wire code. Preserve strict bounded JSON, exact record types and atomic
aggregation; no pickle or opaque unbounded
payload. Numerical instrumentation belongs in a separate coordinated core /
Python change with parity tests and any required paper-first review, not a
transport-local device relabel.

### Consequences

- Positive: distinguishes requested work from observed execution and prevents
  stale/self-declared provenance from satisfying stricter acceptance.
- Positive: permits approved cross-target wheels without conflating their
  archive or installed-member hashes.
- Negative: needs release-policy distribution, numerical-owner observations
  and an explicit migration path; current unverified workers cannot meet the
  strict gate merely by updating transport metadata.
- Negative: hashing and observation add costs. Any caching must preserve loaded
  artifact identity and invalidation; it cannot hide a changed extension.

## Confirmation: proposed test matrix

These are planned checks, **not implemented or passing tests**.

| Case | Required strict behavior |
| --- | --- |
| Missing or echoed-only source evidence | Refuse before executor invocation |
| Missing/substituted loaded extension or tampered installed Python file | Refuse readiness; do not trust RECORD/version alone |
| Wrong target, unapproved artifact or stale release cohort | Refuse readiness |
| Approved Linux/x86_64 and macOS/arm64 artifacts of the same release | Positive admission using their separate approved mappings |
| Missing applicable nodes/weights evidence | Refuse strict completion |
| Changed weight/order/shape/dtype/endianness with the same node count | Detect mismatch; no aggregation |
| Non-quadrature family with explicit approved applicability | Positive admission without a fabricated quadrature claim |
| Requested GPU with observed CPU fallback | Record CPU; reject GPU-only policy, accept only explicit CPU-permitted policy |
| Missing device/precision or only adapter-availability evidence | Keep unknown; refuse strict completion |
| Mixed execution/precision presented as a single GPU/f64 path | Detect mismatch; apply explicit mixed-path policy |
| Wrong envelope/input/seed/unit/family binding | Reject entire batch before mutation |
| Incorrect result digest or evidence from another result | Recompute and reject; no mutation |
| Legacy cached success or changed evidence policy | Cannot satisfy stricter replay admission |
| One invalid cached/reclaimed row among valid rows | No partial aggregation |
| Post-fit provenance failure on a bootstrap replicate | Failure, never statistical rejection or redraw |
| Artifact files changed between readiness and execution | Invalidate admission under the immutable-install trust boundary; no arbitrary loaded-code detection claim |
| Documented pre-execution statistical rejection after valid admission | Positive rejection with explicit not-executed/non-applicable observations; no fabricated kernel facts |
| Pre-execution rejection replayed under a different strict policy | Revalidate current-policy binding; old statistical flag alone cannot qualify |
| Same logical unit with a stricter evidence policy | Preserve seed/resample and archived numeric identities, refuse legacy evidence; only explicitly permitted re-execution |

## Open decisions and non-goals

Jointly settle approved-policy provenance/distribution, the minimal
package-safe verifier boundary and release-tooling/runtime separation,
source-identity mapping, schema/capability version,
canonical node/weight encoding and multi-stage applicability, numerical-owner
receipt locations, strict fallback/mixed-path policies, and cache migration.

No new wire fields, numerical instrumentation, dependency, daemon, build or
full-suite run is authorized by this proposal. No model formula, quadrature
counts/defaults, missing-data rule, chunking or statistical rejection behavior
is changed. Wheel installation, real multi-host numerical execution and actual
GPU evidence remain independent acceptance work.

## Review checkpoint

Valkey worker owner owns this producer/worker boundary proposal; #2286 owner
owns consumer requirements. Lead and Coordinator review the proposed boundary
and negative matrix before an implementation plan. Numerical-core and
release-evidence owners must review instrumentation, approved-artifact mapping
and trust boundaries before any wire implementation. The L4 lead (`fsw-46`)
explicitly owns release-evidence review; the Coordinator
(`fmls-2001-l3-cursor-a6`) explicitly accepted numerical-core review. Their
read-only findings checkpoint is 2026-09-30 14:00 KST. These assignments do not
approve the proposal or implementation. Under the observed high host load,
this phase is read-only tracing and documentation only.
