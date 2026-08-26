# ADR-0022: Rust-owned finite-population proportion sampling design

Status: **Proposed**
Date: 2026-08-26

## Context

Downstream products need auditable sample sizes for finite populations and
stratified reviews. Neither `fast-mlsirm` nor TEPP currently exposes a reusable
artifact for this calculation. TEPP owns temporal-relational documentary
measurement and terminal analysis runs; a general survey/quality-review design
does not require TEPP clocks, ontology, persistence, or service state.

`fast-mlsirm` PRD-PRN-002 and ADR 0001/0002 already assign reusable measurement
arithmetic to its Rust core and allow Python only to validate, marshal, and
present the result. NIST defines the normal-approximation proportion sample
size and finite-population correction for sampling without replacement. The
Australian Bureau of Statistics (ABS) distinguishes proportional allocation
from variability-sensitive optimum allocation and requires independently
selected strata.

## Decision

`fast-mlsirm` owns the versioned
`fast-mlsirm.sampling-design.v1` Rust/PyO3 contract.

The returned immutable artifact retains every canonical numeric input and the
computed result. Rust emits SHA-256 identities for the canonical input and
output encodings and the exact compiled Rust source file, then binds those
identities with the schema, stable source identity, and algorithm version into
one artifact SHA-256. A consumer can
recompute the design from the retained inputs and compare the complete result;
caller-authored hashes are neither accepted nor trusted. Sample-frame identity
and selected-unit membership remain caller-owned provenance and are not inputs
to the measurement arithmetic or its content identity.

This artifact proves only the declared sample-size, FPC, and allocation
calculation. It does not attest selected membership, an achieved estimator,
variance estimate, confidence interval, semantic-coverage result, or permission
for corpus inference; those require separately bound downstream evidence.

For a caller-declared two-sided confidence level `1 - alpha`, absolute margin
`e`, and prior- or pilot-derived proportion `p`, Rust computes

```text
n0 = z_(1-alpha/2)^2 p(1-p) / e^2
n  = ceil(N n0 / (N + n0 - 1))
FPC = sqrt((N - n) / (N - 1))
```

with `FPC = 0` for a census. The overall `p` is the population-weighted value
derived from the declared disjoint strata, so the caller cannot provide a
contradictory aggregate.

The v1 allocation rules are:

- `proportional`: mathematical weight `N_h`;
- `neyman`: equal-cost optimum weight `N_h sqrt(p_h(1-p_h))`.

Integer counts use deterministic largest-remainder apportionment after any
stratum whose mathematical quota reaches its population is made a census and
the remaining formula is reapplied. Input order breaks exact fractional ties.
This is integerization of the declared allocation formula, not an inferred
weight. A design that cannot assign at least one sampled unit to every declared
stratum fails closed; callers must change precision or revise the stratum
design rather than receive an invented minimum.

There is no default `p=0.5`, confidence, margin, design effect, response-rate
inflation, minimum cell count, or cost weight. Although NIST and ABS describe
`p=0.5` as conservative when no prior value exists, using it is a caller policy
decision and must be explicit evidence. Cluster/multistage design effects,
unequal costs, nonresponse, exact-binomial precision, and multi-variable
optimization are outside v1.

## Invariants and acceptance evidence

- Every population count is a positive integer no larger than `2^53`; stratum
  counts sum exactly to `N`.
- Every stratum proportion, confidence level, and margin is finite and strictly
  inside `(0, 1)`.
- Rust owns the inverse-normal quantile, FPC, sample-size, variance term,
  allocation weights, census-cap redistribution, rounding, totals, canonical
  identity encodings, and content hashes.
- Python performs callback-free type/range admission and result marshalling
  only; it contains no sampling formula.
- The allocation sums to `n`, never exceeds a stratum population, and never
  silently drops a declared stratum.
- Tests cover a known finite-population result, distinct proportional/Neyman
  allocations, invalid prior evidence, population mismatch, and infeasible
  stratum coverage.

## Alternatives considered

- Python-side sample-size and allocation arithmetic was rejected because ADR
  0002 assigns ordinary production measurement arithmetic to the Rust core and
  Python is limited to validation, marshalling, orchestration, and reporting.
- Exact-binomial sizing was not substituted for v1 because it is a different
  method with different guarantees and requires an explicit separately
  versioned scientific contract rather than a silent formula change.
- Unequal-cost optimum allocation was excluded because v1 has no governed cost
  evidence contract; inventing or defaulting cost weights would change the
  design without caller evidence.
- An implicit conservative `p=0.5` was rejected because the expected
  proportion is caller-declared prior or pilot evidence and the package must
  not manufacture that evidence.

## Consequences and trade-offs

The contract is independently usable by any Python/Rust consumer without TEPP
service coupling. Its normal approximation is not an exact-binomial guarantee,
power calculation, or complex-survey optimizer. Consumers must show the
declared prior/pilot provenance and must not call `n` a product-wide evidence
threshold.

## Compatibility, rollback, and reversal

Unknown schema versions fail closed. A future method adds a new schema major or
an additive method identifier with its own ADR and verification; it does not
silently change v1 results. Removing the public wrapper rolls back the candidate
without a database migration because the artifact is pure and stateless.

## References

Australian Bureau of Statistics. (2023, March 1). *Sample design*. https://www.abs.gov.au/websitedbs/D3310114.nsf/home/Basic%20Survey%20Design%20-%20Sample%20Design

National Institute of Standards and Technology. (n.d.). *Sample sizes
required*. In *NIST/SEMATECH e-Handbook of statistical methods*. Retrieved
August 26, 2026, from
https://www.itl.nist.gov/div898/handbook/prc/section2/old.prc272.htm

National Institute of Standards and Technology. (n.d.). *Confidence limits*.
In *NIST/SEMATECH e-Handbook of statistical methods*. Retrieved August 26,
2026, from
https://www.itl.nist.gov/div898/handbook/prc/section2/old.prc271.htm
