# ADR-0009: Adaptive rotation uses criterion registry, multi-start and empirical selection

Status: **Accepted**  
Implementation maturity: **Protected-main CPU baseline implemented; GPU/additional-criterion/recovery expansion remains planned**  
Date: 2026-08-09

## Context

Exploratory factor rotation does not have one universally optimal criterion. Criterion behavior depends on loading complexity, factor correlation, cross-loadings, sample size, target information and local optima. Comparing raw objective values across different criteria is also invalid because those objectives have different definitions and scales.

Protected main now contains the Rust-backed rotation criterion registry, deterministic multi-start optimizer, criterion-neutral selector/evidence surfaces, Python/PyO3 public API, package-root rotation exports, method doctoring and regression coverage. The governing policy in this ADR therefore describes current protected-main behavior. GPU batching, additional criterion families and broader simulation/recovery evidence remain future increments and do not make the implemented CPU policy Proposed.

## Decision

Rotation is structured in three layers.

### 1. Criterion registry

Each analytic criterion implements a common Rust value/gradient contract. The optimizer does not contain criterion-specific algebra except through that interface.

The protected-main registry contains orthomax/Crawford-Ferguson, oblimin, geomin, target/PST, information, component-loss, bifactor, tandem and invariant-simplicity families documented by `available_rotation_criteria()`. Procedural or derivative-free criteria may use explicit separate adapters rather than being forced into an invalid gradient contract.

### 2. Optimizer and solution search

The Rust optimizer supports appropriate orthogonal/oblique geometry and reports:

- criterion value;
- projected gradient/stationarity;
- transform/pattern/structure/factor-correlation matrices;
- termination reason;
- deterministic multi-start evidence;
- best-start index and best-observed basin support;
- sign/permutation canonicalization where semantically valid.

Finite multi-start returns the **best observed solution**, not proof of a global optimum.

### 3. Criterion-neutral selector

Criteria are compared using common evidence such as:

- loading/simple-structure complexity;
- cross-loading sparsity;
- degeneracy/near-singular factor-correlation penalties;
- convergence and basin support;
- bootstrap stability and Tucker congruence after global assignment/sign alignment when replicates are supplied;
- target/theory agreement when externally provided; and
- split-sample or simulation recovery when the study design supplies it.

The selector exposes explicit evidence policies, but policy weights/ranks are documented choices, not universal scientific constants. Objective values from unlike criteria are never treated as directly comparable merely because they are scalar.

## Numerical constraints

- Matrix operations used by a criterion must respect its mathematical domain; SPD log determinants use SPD-safe decomposition such as Cholesky rather than sign-ambiguous row pivot heuristics.
- PST/target weights follow the exact documented criterion semantics. Binary-mask PST does not silently accept arbitrary continuous weights.
- Hyperparameters are included in solution provenance.
- CPU multi-start uses coarse parallelism to limit context switching.
- GPU batching is released only after objective/gradient/stationarity and selected-basin parity evidence; current protected-main rotation provenance identifies the CPU backend rather than implying GPU execution.

## Invariants and current acceptance evidence

The Accepted baseline is evidenced by protected-main source/tests/docs including:

- `crates/mlsirm-core/src/rotation/` for criteria, optimizer, matrix helpers and selector;
- `crates/fast-mlsirm-py/src/rotation_bindings.rs` for the native binding;
- `python/fast_mlsirm/rotation.py` and `rotation_selection.py` for public validation/marshalling/report surfaces;
- package-root rotation exports;
- `tests/test_rotation*.py` for public API, selection, validation and target-alignment contracts; and
- `docs/adaptive_factor_rotation.md` for the current scientific/operational boundary.

Future criteria, GPU execution and stronger population-recovery studies require their own exact-head parity/recovery evidence before their claims become implemented. Their absence does not invalidate the current criterion-neutral CPU architecture.

## Consequences

The public product can answer both “what is the best solution we observed for this criterion?” and “which criterion is best supported for this use case?” without conflating the two. The second answer remains conditional on candidate set, extraction model, data/resampling design and policy; it is not a universal criterion-ranking claim.

## Failure and interpretation boundaries

- Non-convergence, singular/invalid domains, unsupported criterion/mode combinations and invalid target/weight semantics fail explicitly rather than silently changing objective meaning.
- A finite set of starts cannot certify the global optimum.
- Bifactor-oriented rotations do not establish substantive bifactor scoreability or justify a general score by themselves.
- Criterion selection evidence does not replace factor-retention, structural-model comparison, held-out prediction, invariance/DIF or true-structure recovery when those questions govern interpretation.

## Alternatives considered

- **Default to varimax.** Rejected as universal policy.
- **Choose the criterion with the numerically smallest objective.** Rejected because objectives are not cross-criterion comparable.
- **Use one random start.** Rejected because local minima can dominate results.
- **Claim finite multi-start global optimization.** Rejected as scientifically unsupported.

## Planned evolution without changing this Accepted decision

- GPU batching after parity and basin-selection evidence;
- additional scientifically justified criterion families;
- richer bootstrap/simulation recovery evidence and buyer-facing selection reports; and
- supersession through a later ADR if the registry/optimizer/selector ownership model itself changes.

## References

Bernaards, C. A., & Jennrich, R. I. (2005). Gradient projection algorithms and software for arbitrary rotation criteria in factor analysis. *Educational and Psychological Measurement, 65*(5), 676–696. https://doi.org/10.1177/0013164404272507

Browne, M. W. (2001). An overview of analytic rotation in exploratory factor analysis. *Multivariate Behavioral Research, 36*(1), 111–150. https://doi.org/10.1207/S15327906MBR3601_05
