# ADR-0008 — Statistical Evidence, CI Tiers and Release Gates

- **Status:** Accepted
- **Date:** 2026-08-09

## Context

Psychometric software needs expensive recovery and simulation studies that can
consume substantially more time than normal PR validation. Running every heavy
study on every commit can saturate CI and slow defect repair; removing those
studies to improve latency weakens scientific evidence. Monte Carlo thresholds
can also become invalid if tuned after observing one seed's result.

## Decision

Use tiered validation while preserving one scientific release contract:

### Per-PR required evidence

- normal Python/Rust/PyO3 tests;
- statement/branch/public-docstring gates;
- numerical delegation/parity relevant to the change;
- bounded realistic recovery/smoke evidence where material;
- package/reinstall/public API checks;
- explicit GPU no-skip for changed GPU contracts;
- fuzz/security/SAST/supply-chain gates;
- documentation/changelog/contract consistency.

### Heavy statistical studies

Long Monte Carlo, paper-design recovery and broad condition grids may run as
scheduled, manual or release workflows with immutable exact-source identity,
bounded shards/timeouts and machine-readable artifacts. A release claim that
depends on those studies requires fresh accepted evidence on the release
contract; the evidence cannot disappear because it is not in every PR job.

## Statistical acceptance rule

Acceptance criteria are specified from methodological requirements, desired
precision/power/coverage or a documented historical compatibility contract before
using the target outcome where feasible. A threshold MUST NOT be loosened solely
because one deterministic seed failed it.

Finite Monte Carlo uncertainty is represented explicitly. For example, an
observed convergence proportion is a binomial estimate and should not be compared
to a population target as if it had zero sampling error unless the requirement
really is an exact deterministic threshold.

## Invariants

- ignored/slow test inventories are exhaustive and disjoint; no scientific test
  silently disappears from all shards/workflows.
- empty or skipped evidence cannot be promoted to success.
- CI concurrency may cancel superseded runs but never an exact current-head gate
  that is still needed for merge/release proof.
- exact-head, merge-ref and release-artifact identities are distinguished.
- queued/pending/cancelled/predecessor/synthetic/status-only evidence is not pass.
- self-modifying PR-controlled workflows do not generate and push reviewed source.

## Alternatives considered

1. Run every recovery grid on every push — rejected as unnecessary queue
   saturation.
2. Remove heavy studies — rejected as scientific regression.
3. Tiered PR vs scheduled/release evidence with an explicit final release gate —
   accepted.

## Consequences

PR latency can remain bounded while long-running scientific evidence continues.
Release automation must know which heavy studies apply to the changed scientific
surface and bind their artifacts to the exact source/release.

## Failure / degraded behavior

Infrastructure or reviewer latency defers only the affected action. A failed
scientific study is a substantive failure until RCA identifies a code/model/test
problem; it is not dismissed as CI flakiness without evidence.

## Security / supply chain

Study runners use bounded subprocess/process-group termination, pinned/verified
inputs where required, no model secrets unless explicitly necessary, and
artifact integrity hashes. PR-controlled code never receives unnecessary
write/admin credentials for scientific validation.

## Verification

- shard inventory tests;
- timeout/process-group tests;
- exact-source manifest tests;
- known-parameter bias/RMSE/coverage/convergence studies;
- CPU/GPU parity;
- release acceptance that consumes the actual wheel/source artifact;
- changelog and evidence-index parity.

## Supersession criteria

Supersede if CI capacity and deterministic scientific study design make a simpler
tier system equally fast and more reliable without losing release-bound recovery
evidence.
