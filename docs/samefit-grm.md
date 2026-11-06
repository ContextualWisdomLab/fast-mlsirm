# Same-fit multigroup GRM diagnostics

## Scope and method

`samefit_grm::import_summary` accepts complete response data and a caller-selected,
separately authenticated result summary. It verifies request SHA-256, item identity,
common/free coordinate order, quadrature count, and terminal marginal log likelihood
using the caller's explicit absolute tolerance. It does not optimize, establish
convergence, regenerate a historical identity, or prove historical decoded-array
bit equality. `PackedArchive` preserves binary64 packed coordinate bits. Signed
slopes are supported; ordered intercepts use a first intercept and positive
exponentiated gaps. Reference group mean/SD are fixed at zero/one.

`samefit_grm_joint` uses all cumulative univariate indicators and cross-item
cumulative bivariate indicators, group-major. Its analytic tangent includes actual
common/free item coordinates and focal-group means/log standard deviations.
This is the full first/second-order moment construction, not M2* or C2.
Maydeu-Olivares (2013, p. 78, Equations 10–11) supplies the quadratic-form and
tangent-projection basis. Cumulative indicators are an invertible linear change
of the nonredundant category-indicator basis; covariance and tangent use that same
basis. For independent groups, covariance blocks are `(N / N_g) * Xi_g`, and
projection multiplies the residual quadratic form by total `N`.

`samefit_strict::project` performs unregularized Cholesky whitening, normalized
column pivoting and two-pass modified Gram–Schmidt. Rank uses machine epsilon
times the larger matrix dimension. Returned df is moment count minus measured
rank. Non-positive-definite covariance is rejected: no ridge, pseudoinverse, or
CPU fallback is used to rescue a failed calculation. Pivot diagnostics are
numerical diagnostics, not a finite-sample calibration certificate. The joint
consumer delegates df admission to this measured rank, including when packed
coordinates are redundant. Positive algebraic df at a nonidentified parameter
point does not establish the regular-model chi-square reference distribution.
Finite tangent columns are max-absolute scaled before whitening and normalized
with a scaled sum of squares, without changing the rank cutoff.

The result-file consumer requires repeated identities to agree in packed values,
coordinate map, anchor mask, quadrature and terminal likelihood before selecting
a summary. Fit-index calculation rejects nonfinite derived N*df denominators,
p-values and noncentral limits, or reversed limits, before JSON serialization.
Undefined comparative indices retain their separate nullable contract.

## Numerical backend and limitations

With the `gpu` feature, `joint_diagnostics` requires GPU f32 moments/covariance.
`joint_diagnostics_with_backend(..., MomentBackend::Cpu)` explicitly selects CPU
f64 instead. Neither selection is controlled by ambient environment variables.
Tangent, projection and distribution calculations are CPU f64 in both cases.
GPU output includes actual device and dispatch metadata and CPU parity errors.
The inherited absolute moment-parity refusal is `5e-6`; it is an implementation
check, not a scientific fit threshold or assurance of covariance accuracy.
GPU covariance sensitivity must be evaluated for the caller's application.
The low-level archive/import, tangent, projection and index APIs do not need GPU.

The independence comparator is newly constructed from each group's empirical
cumulative margins. It is not a recovered historical null fit. Boundary margins
or singular covariance refuse calculation. CFI/TLI can be null when the comparator
has no positive excess discrepancy. The first-threshold indicator correlation
residual RMS weights group/item-pair cells equally. It is not full ordinal-score
Pearson or polychoric SRMR: compare Maydeu-Olivares (2013, pp. 84–85, Equations
34–36), which use the ordinal item values themselves.

## RMSEA denominator

Same-fit RMSEA is `sqrt(max(M2 - df, 0) / (N * df))`. Its 90% interval divides
noncentrality limits by the **same `N * df`** before taking square roots. The
limits solve chi-square CDF targets 0.95 and 0.05, respectively
(Maydeu-Olivares, 2013, p. 82, Equations 26–28). Output labels this denominator.
Existing single-group `fitstats` paths use **`(N - 1) * df`**; their definition is
not changed here. Do not compare outputs as if those denominators were identical,
and do not pass `N - 1` as sample size to conceal the convention.

## API boundary and verification

The importer is the shared dependency for downstream same-fit parameter consumers.
Consumers should use `ImportedFit::archive`, `responses`, `groups`, `quadpts` and
`item_ids`, rather than introduce another importer. `import_summary` does not
verify the enclosing result-file digest: the example verifies it before selecting
a summary. Callers must do the same and supply trusted expected hashes; hashes
alone do not authenticate an untrusted producer.

Synthetic tests cover packed-coordinate round trips/refusals, SHA/map/likelihood
tampering, signed slopes, analytic derivatives, GPU enumeration parity, unequal
group weighting, duplicate tangent rank, singular covariance refusal, scale
invariance and noncentral interval ordering. They do not establish fit-estimation
convergence or finite-sample nominal rejection rates for an arbitrary application.
No application data, fit identities, private numerical results or local file
locations are part of these fixtures.

## Reference

Maydeu-Olivares, A. (2013). Goodness-of-fit assessment of item response theory
models. *Measurement: Interdisciplinary Research and Perspectives, 11*(3),
71–101. https://doi.org/10.1080/15366367.2013.831680

The cited full-text passages are read for this implementation. Only the citation,
link and this original summary are distributed; PDF redistribution permission is
not established.
