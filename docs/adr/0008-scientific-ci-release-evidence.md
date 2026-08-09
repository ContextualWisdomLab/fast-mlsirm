# ADR-0008: Separate bounded PR evidence from heavy scientific studies and release proof

- Status: **Accepted**
- Date: 2026-08-09
- Owner: scientific CI/release governance

## Context

Psychometric software needs stronger evidence than ordinary unit tests: parameter recovery, Monte Carlo convergence/coverage, CPU/GPU parity, factor/model-selection recovery, and realistic domain studies can be expensive. Running every large study on every pull request can exhaust runners and slow defect correction; removing the studies weakens scientific assurance.

A second recurring failure mode is reporting correlation between true and estimated values as if it proved parameter recovery. Correlation preserves rank under many biased linear transforms and does not establish scale accuracy, uncertainty coverage, or decision calibration.

## Decision

Scientific evidence is tiered:

### Pull-request gate

Run bounded deterministic tests that can expose formula/API/resource regressions quickly: focused recovery fixtures, Rust/Python parity, owned coverage/docstrings, GPU no-skip smoke where relevant, security/SAST/fuzz/package checks, and representative numerical edge cases.

### Scheduled/manual scientific studies

Run expensive Monte Carlo, large true-parameter recovery, broad CPU/GPU parity, stress/fuzz, factor/model-selection simulations, and benchmark studies with explicit environment/seed manifests.

### Release gate

A release that changes scientific behavior must link the exact protected head/artifact to the relevant completed study evidence. Release acceptance also requires package/reinstall, compatibility, security, coverage, SBOM/provenance, and independent review evidence.

## Invariants

1. PR smoke is bounded but does not replace the corresponding heavy study when the release claim needs it.
2. Heavy studies are not tuned to observed seed outcomes. Monte Carlo acceptance rules are justified prospectively by the scientific target and sampling uncertainty.
3. True-parameter evidence uses bias, MAE/RMSE, interval/SE coverage, convergence, response/information recovery, and DIF/invariance as applicable; correlation is supplementary only.
4. Latent parameters are aligned to an identified/common scale before componentwise recovery metrics are computed. Multidimensional/latent-space comparisons handle rotation, sign, permutation, or Procrustes invariance.
5. GPU evidence must prove an actual GPU execution, not a software/CPU fallback described as GPU success.
6. A failed required check/study remains failed; reruns are appropriate only for demonstrated transient/infrastructure causes.
7. A release is not cut from predecessor-head, synthetic-only, stale-base, skipped-required, or missing evidence.

## Alternatives considered

- **Run all studies on every PR:** rejected because queue exhaustion itself reduces product reliability and does not add proportional evidence for small changes.
- **Run only unit tests:** rejected because estimator/model-selection claims require population/recovery evidence.
- **Use correlation as the recovery criterion:** rejected because it cannot detect offset/scale bias or uncertainty miscalibration.
- **Accept any CI green aggregate when a required subcheck is skipped:** rejected because unexecuted evidence is not passing evidence.

## Failure and recovery

PR failures produce the smallest actionable boundary; source defects are fixed test-first. Heavy-study failures block the affected scientific/release claim but do not freeze unrelated development. Transient runner/provider failures are classified and retried within a bounded policy; deterministic scientific failures require a source/design correction.

## Compatibility and rollback

Changing a study design, acceptance threshold, seed policy, recovery metric, or release evidence requirement is a governed scientific change and must preserve historical result interpretability. A release rollback retains the original study/provenance artifacts.

## Verification

CI tests shall inventory intended ignored/scheduled studies, prevent empty/duplicate shards, verify subprocess deadlines and cleanup, require exact-head artifact identities, and exercise release-acceptance scripts.

## Consequences

The policy trades some immediate PR completeness for faster trustworthy iteration while retaining rigorous evidence at the appropriate scheduled/release boundary.
