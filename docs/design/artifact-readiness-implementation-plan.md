# Artifact readiness implementation plan

Status: **Proposed plan, not implementation approval**. Date: 2026-09-30.
Architectural baseline: reviewed proposal at
`5447445ad5129d6638877645f9d31b962f778ee4`; keep that document frozen as reviewed.
Core, consumer and release **design findings are resolved**. The decisions
below are a narrow implementation proposal, not pending consumer review.

## Motivation and scope

First implement local artifact-file readiness without changing numerical
outcomes, remote wire records or caches. A trusted independent artifact policy
must be checked against actual installed files; source/version strings or a
producer-supplied bundle cannot approve themselves. Numerical stage evidence
requires a separate core instrumentation change.

## Tranche A: artifact readiness only

Proposed footprint: one stdlib-only package module
`python/fast_mlsirm/_artifact_readiness.py` and one focused test file
`tests/test_artifact_readiness.py`. No CI-script imports, build-tool pin,
generic plugin/factory, signature service or numerical import in the helper.
Do not change `remote_exec`, `remote_worker`, public fit functions or wire/cache
behavior in this tranche.

One host-facing evaluation operation takes an independently configured policy,
its separately pinned digest, an installation root and trusted startup
observations. None of these approval inputs may come from a job record.
Missing approval or required startup facts returns explicit unknown readiness;
contradictory known facts return mismatch. Matching files under the stated trust
profile yields **artifact readiness only**, never numerical execution evidence.
Malformed policy data raises a documented validation error before file access.

### Proposed policy fields

Policy schema identifier: `worker_artifact_policy/1.0`. Exact field names and
limits below require joint approval before code.

| Field | Meaning |
| --- | --- |
| `schema` | Exact supported policy identifier |
| `distribution` | Exact package name/version; no coercion or version-only approval |
| `source_commit` | Independently approved 40-hex source commit, not an installed-file or archive SHA-256 |
| `installation_profile` | Host-owned immutable-install profile; not a job-controlled boolean |
| `targets` | Approved exact platform/machine/Python-ABI descriptors |
| Each target's `archive_sha256` | Approved distribution digest, not reconstructed from installed files |
| Each target's `extension_member` | Approved relative numerical-extension member |
| Each target's `members` | Sorted unique relative paths, exact byte sizes and lowercase SHA-256 values |

### Exact proposed nested shapes and API

All objects are closed exact-key mappings when parsed; lists have the bounds
above and strings are exact built-ins, never caller-coerced objects.

```text
Policy = {schema, distribution, source_commit, installation_profile, targets}
  distribution = {name: "fast-mlsirm", version: bounded string <=64 bytes}
  installation_profile = "host_immutable_install/1.0" (other profiles => unknown)
  targets = sorted list of Target, unique target_id and descriptor
Target = {target_id, descriptor, archive_sha256, package_roots,
          extension_member, members}
  descriptor = {sys_platform, machine, python_abi}: bounded strings <=128 bytes
  package_roots = sorted unique relative POSIX directories (1..16, nonoverlapping)
  members = sorted list of {path: relative POSIX path, size_bytes: exact int,
                            sha256: lowercase 64-hex string}
StartupObservation = {distribution, descriptor, loaded_origin, worker_pid,
                      startup_policy_digest, startup_installation_epoch,
                      current_installation_epoch, immutable_profile}
  distribution/descriptor = same closed shapes (unknown scalars may be None)
  loaded_origin = trusted caller's absolute imported-extension origin or None
  worker_pid = exact positive int or None
  startup_policy_digest = lowercase 64-hex or None
  epochs/profile = bounded strings <=128 bytes or None
ArtifactReadiness = {state, reason_code, scope, artifact_policy_digest,
                     target_id, source_commit, archive_sha256,
                     observed_members_sha256, worker_pid, installation_epoch}
  state = ready | unknown | mismatch; reason_code = closed bounded code
  scope = artifact_files_only; unknown identity fields are None
```

Proposed Python operation:

```text
verify_artifact_readiness(policy_json: str | bytes | None, *,
    pinned_policy_digest: str | None, installation_root: Path | str,
    startup_observation: dict | None, deadline_monotonic: float)
    -> ArtifactReadiness
```

The startup record is supplied by a **trusted caller**, not authenticated by
this helper. Its target/process/startup-policy/epoch facts must match current
runtime and independently configured approval. A forged caller passing identical
trusted values is outside this trust model. Hash equality cannot establish policy
authority origin: identical policy bytes matching an independently approved host
pin are not rejected because an untrusted sender transported those bytes. A job
cannot set/replace the host pin or startup profile; malformed override fields
and content differing from that pin are rejected. A03 tests this configuration
boundary, not magical producer-origin detection.

Policy digest: SHA-256 over a domain tag
`fast-mlsirm/artifact-policy/1` plus canonical strict JSON of the policy body.
Use UTF-8, sorted object keys and member ordering, compact separators and no
nonfinite values. The body does not contain its own digest. Compare to the
independently pinned host digest; computing a digest does not approve the body.
Approve different target archives/members for the same release explicitly.
Do not reinterpret the existing numerical manifest's 64-hex source declaration
or derive a source commit/wheel archive hash from installed-member hashes.

### Proposed bounds and file checks

- Policy JSON: at most 1 MiB UTF-8 and depth 64; reject duplicate keys, unknown
  fields, nonfinite constants, bool-as-int and primitive subclasses/coercion.
- At most 16 targets and 4,096 members across the policy; path at most 512 UTF-8
  bytes and ordinary identifiers at most 128 bytes. Digests have exact lengths.
- Selected installed members total at most 1 GiB, with exact nonnegative integer
  sizes. Stream hashing in 64 KiB chunks. These reuse the existing release
  inventory byte/I/O scales, not its CI environment assumptions.
- Paths are canonical relative POSIX members: no absolute/drive/UNC paths,
  traversal, backslashes, control characters, file/network URIs or executable
  loading. Resolve only beneath the trusted installation root. Reject links,
  special files, duplicate/aliased members and unapproved extension origins.
- Selected member-set closure is exact under every declared package root:
  every regular file must be a declared approved member, including resources,
  metadata and generated bytecode if present. **No `__pycache__`, `.pyc`, native
  or Python-source directory/file is implicitly ignored.** Metadata outside the
  declared roots is outside verification scope, not proof of approval. An
  unlisted generated file refuses readiness; the helper never auto-approves it.
  This conservative first-tranche choice avoids an unsafe generated-cache
  exception. Bound visited entries to 8,192 and traversal depth to 64.
- Retain an installation-root descriptor and use descriptor-relative,
  no-follow directory/file opening where supported. Verify regular-file identity
  on the opened descriptor; compare pre/post `fstat` identity, size and mutation
  metadata, then confirm the path still names that descriptor. Changed files
  refuse readiness. If safe opening cannot be supported, return unknown instead
  of following a link or presenting a check-then-open path check as protection.
- Verify target, exact selected member set, actual regular-file sizes/hashes and
  trusted startup extension-origin mapping. Never import or execute a candidate
  file to verify it. Require a finite absolute monotonic caller deadline and
  check it before operations and between 64 KiB reads; a reached deadline returns
  unknown. This is cooperative: a blocking filesystem call cannot be preempted
  by that check. The caller must also supervise a hard process deadline and
  installation lifetime. A 1 GiB cap is not a wall-clock guarantee. The helper
  creates no daemon, process or package installation.

Readiness facts bind to the artifact-policy digest, selected target, observed
member inventory, worker process and host installation epoch/profile. A missing
loaded-extension observation or unknown immutability profile stays unknown.
Any installation/startup-policy change invalidates the observation; require a
new trusted startup/admission rather than accepting a refreshed path hash as
proof of already loaded bytes. File hashes do not establish immutable loaded
Python/bytecode/native code against a malicious worker or privileged host.
The host-approved installation/load profile must state that trust boundary.

Reuse verification **semantics** from `bundle_inventory:92`,
`verify_runtime_inventory:263` and installed-file capture `:27`; do not import
the tooling or require `uv 0.12.5`, release-job `build_env` or a complete release
receipt in a generic live worker. Keep source commit, archive digest and actual
member observations distinct. The output is a bounded package-owned readiness
record; it has no node/device/precision or numerical-completion claims.

### Planned RED inventory A (not written or executed)

| ID | Behavior pinned first |
| --- | --- |
| A01 | Missing independently pinned approval returns unknown before file access |
| A02 | Unknown policy schema, malformed/extra/duplicate/nonfinite data or bool sizes cannot qualify |
| A03 | Changed policy body or attempted job override cannot replace the independent host pin/profile; identical approved bytes are not rejected by guessed origin |
| A04 | Wrong/unapproved target, interpreter ABI or release source mapping refuses readiness |
| A05 | Missing declared member, wrong size or swapped file bytes refuses readiness |
| A06 | Missing/unapproved/out-of-root extension observation cannot qualify |
| A07 | Traversal, URI, link, special-file, duplicate/alias or any unlisted member (including bytecode) is rejected without execution |
| A08 | Policy bytes/depth/member count/path/selected-byte budget is bounded before unbounded work |
| A09 | Changed epoch/startup policy, descriptor/path identity or file during hashing invalidates readiness; timeout is unknown, not loaded-byte proof |
| A10 | Unknown mutable-install profile stays unknown despite matching hashes |
| A11 | Artifact-ready output contains no fabricated numerical receipt or numerical capability |
| A12 | Positive: independently approved cross-target mappings match separate fixture files/digests |

Use tiny temporary files and trusted-startup fixtures, not new wheels, actual
native-module loading, fit calls or numerical kernels. These test file/record
validation under an explicit trust assumption, not real installed-wheel/GPU
acceptance. Avoid platform-dependent skip allowances; unsafe-file cases can use
bounded metadata fixtures where filesystem privileges differ.

After plan approval, run each relevant test to establish RED, implement the
smallest verifier, then run this focused test file to GREEN under a resource
window. Do not label the inventory as failing tests before running it. Preserve
all existing numerical/transport code in this tranche.

## Follow-up B: additive receipt and strict admission

Separate plan/approval gate; **not part of tranche A implementation**.
Candidate receipt identifier: `worker_execution_receipt/1.0`. Keep its
serialization Proposed until core stage/byte encoding and completeness profiles
are agreed. No opaque arbitrary observation blob or transport-invented stage.

Bind a real receipt to exact logical envelope fingerprint, run/unit/family,
existing input identity, separately recomputed payload identity, original
derived seed and recomputed result digest. Store a receipt only if it actually
exists, separately from immutable archived schema-1 outcomes/result hashes.
Artifact readiness is not a whole-numerical-execution receipt.

The strict-admission policy has its own domain-tagged digest and explicitly
references the artifact-policy digest, required receipt schema and observation
profile. Do not conflate these two policy scopes. Candidate admission key:
SHA-256 of domain `fast-mlsirm/evidence-admission/1` plus canonical JSON containing
logical envelope fingerprint, strict-policy digest and receipt-schema ID.
These values are **never RNG inputs**. Validate receipt/result binding before
using that key; archive old numerical identities without rewriting them.

Strict mode is explicit opt-in. Legacy readable results with missing/inadequate
receipts return evidence-inadequate: zero implicit executor calls, redraws or
resampling. A separately authorized re-execution retains the original logical
unit/seed and records a new observation; it does not overwrite the archive or
attach a new result's receipt to a different archived result. Unknown receipt
schema or required API capability is inadequate before fit when detectable.
Until the core provides applicable actual observations, numerical strict
requirements remain unsatisfied even when artifact files match.

Keep the complete future logical transport message at most 1 MiB UTF-8/depth64;
propose a 128 KiB receipt sub-budget **inside** that total, not an extra allowance.
Code-grounding caveat: today's `_valkey_json:777` checks one JSON field at a
time; those checks alone do not prove a combined-record cap. Follow-up integration
must define/count the whole canonical framing plus outcome/receipt and test
that aggregate bound. Do not increase existing caps or split unbounded evidence
across chunks. Validate the whole batch and replay bindings before aggregation
or strict admission mutation; raw legacy archival is not strict acceptance.

### Planned RED inventory B (not written or executed)

B01 legacy-readable/strict-refused; B02 same unit/new policy preserves seed and
sample; B03 unknown schema/capability; B04 tampered policy/envelope/input/seed /
result bindings; B05 inadequate cache hit causes zero executor calls and no
archive mutation; B06 one invalid row leaves the entire aggregate untouched;
B07 receipt sub-budget/combined message/depth/type violations; B08 readiness-only
or fabricated receipt never upgrades numeric evidence. Positive same-policy
replay requires a genuine supported receipt, not an invented profile. Numerical
stage observations and legitimate not-executed cases remain governed by the
reviewed architecture and deferred core instrumentation.

## Non-goals and alternatives

No numerical instrumentation, formula/default/node-count/tolerance change,
remote wire/cache edit, new transport/client dependency, auth/signature service,
daemon, build or full-suite run now. Importing CI helpers wholesale is rejected
because release-job assumptions are not runtime policy. Hash-only self-approval
or readiness presented as numerical evidence is also rejected.

## Approval and KST checkpoints

Release reviewer: Lead `fsw-46`; core reviewer: Coordinator
`fmls-2001-l3-cursor-a6`; consumer reviewer: `fmls-2001-l4-durable-dedup-f6`.
The reviewed architecture remains Proposed while these concrete policy fields,
bounds, readiness trust/API boundary and follow-up migration/schema are settled.
The first plan was delivered at 2026-09-30 14:09:35 KST, before its 14:10
checkpoint. The 14:30 disposition target slipped while reviewers required
concrete nested shapes, authority-origin limits, file-set closure and safe
opened-file/mutation checks. These are now explicit, rather than called review
blockers without a fix.

At 15:26 KST: load 4.29 on 10 CPUs, memory-free indicator 38%, swap headroom
approximately 1.58 GiB; Podman remains capped at 2 GiB. Revised checkpoints:
plan clarifications by 15:45, joint scoped-A confirmation by 16:00, focused RED
by 16:15 and GREEN by 16:45 **only after confirmation and a fresh resource
check**. Adjust from actual results; do not assume approval or completion.
Implementation belongs on a separate artifact-readiness branch, not the #2283
transport ABI. B remains unapproved; do not let its unresolved numerical schema
block independently reviewed A. No implementation, RED execution, wire edits
or build starts from this plan alone. The architecture's 24 planned cases remain
separate from this inventory.
