# ADR-0022: Rust-owned polytomous period artifact

Status: **Proposed**
Date: 2026-08-26

## Context

Downstream products compare groups and periods on a shared GRM/GPCM metric.
One current consumer builds response matrices, summarizes EAP scores, ranks
item information, selects between GRM and GPCM, and assembles residual-map
evidence in Python. That duplicates production psychometric arithmetic and
allows a product-specific selection score to become measurement policy.

GRM and GPCM are not made nested by sharing ordered response categories.
ADR-0006 and PRD-FR-042 require formal distinguishability evidence before a
non-nested preference. A weighted combination of likelihood and mean outfit is
not such evidence, and no arbitrary coefficient may be promoted into this
contract.

## Decision

`fast-mlsirm` will own one versioned, domain-neutral
`polytomous_period_artifact` Rust computation. The request contains:

- unique opaque person and item identifiers;
- observed `(person_id, item_id, category)` cells;
- declared category count and bounded numerical configuration;
- `free_calibration` or `fixed_item_bank` mode;
- for fixed mode, an ordered content-addressed item bank; and
- an optional prior-period mean used only to compute an explicit difference.

Rust maps identifiers to a missing-aware response matrix and rejects duplicate
cells, unknown identifiers, invalid categories, non-finite values, unsupported
schema versions, and allocation overflow before fitting. Free calibration fits
the GRM and GPCM candidates through the existing Rust kernels. It returns no
preferred model unless the relation-safe comparison required by ADR-0006 has
established formal distinguishability and the selected candidate has adequate
convergence evidence. `indistinguishable`, `requires_distinguishability_test`,
and `no_converged_candidate` are valid terminal analysis states.

The immutable result carries canonical input and artifact SHA-256 identities,
schema/algorithm/implementation versions, candidate convergence and comparison
evidence, the selected item bank when selection is supported, per-person EAP
and posterior SD, Rust-computed population summaries and optional period
difference, item information with deterministic ranking, expected responses,
and the ADR-0021 residual-interaction-map envelope.

Python may validate resource bounds, marshal the request, and expose typed
results. It must not construct the numerical matrix, calculate score summaries
or differences, rank item information, reproduce residual arithmetic, or select
a preferred model.

## Ownership and downstream boundary

The artifact uses opaque identifiers and contains no tenant, authorization,
database, calendar, grouping, or UI policy. A downstream product remains
responsible for authorization, group/period semantics, persistence, display,
and verifying the exact artifact schema, input digest, implementation version,
convergence, and selection state before projection.

## Acceptance evidence

The proposal becomes Accepted only after all of the following land on a
protected head:

1. Rust row-order invariance and fail-closed admission tests for duplicate,
   unknown, missing, non-finite, category, schema, and resource boundaries.
2. Deterministic cross-language canonical fingerprints under ADR-0003.
3. Synthetic GRM/GPCM recovery with bias, MAE, RMSE, interval coverage,
   convergence, and model-selection accuracy—not correlation alone.
4. A formally indistinguishable sample that returns no winner and a
   distinguishable sample whose preference is emitted only after the formal
   test.
5. Fixed-bank two-period recovery proving score-location and mean-difference
   preservation.
6. Exact population-mean/SD/difference and deterministic item-information rank
   tests, including ties.
7. Reuse of the ADR-0021 residual artifact with its rank-zero, incomplete-data,
   reconstruction, coverage, extrema, and resource-bound tests.
8. PyO3 contract tests proving the compiled Rust core is required and an AST
   guard proving the public Python layer contains no replacement arithmetic.
9. A downstream shadow comparison on synthetic data before the downstream
   Python formulas are deleted.

## Failure, privacy, and security behavior

An unidentified relation, failed convergence, incompatible item bank, invalid
cell, digest mismatch, or unavailable Rust core returns an explicit unavailable
state or error; it never falls back to a Python estimate. Artifacts retain only
opaque identifiers and numerical evidence. Source text and identifying product
metadata remain outside the reusable artifact.

## Compatibility and migration

This adds a new `fast-mlsirm.polytomous-period-artifact.v1` contract. It does
not reinterpret existing downstream calibration scores. Consumers must treat
those values as legacy product evidence, make the field nullable where needed,
and migrate only newly produced owner artifacts after shadow verification.
Rollback disables the new consumer path; it does not restore local Python
arithmetic as an automatic fallback.

## Alternatives considered

1. **Port the downstream weighted calibration score to Rust.** Rejected because
   moving an unsupported coefficient does not make the selection rule valid.
2. **Expose only lower-level Rust arrays.** Rejected because downstream mean,
   ranking, matrix, and residual assembly would remain a second production
   numerical implementation.
3. **Create a hosted measurement service.** Rejected because the existing
   package/PyO3 boundary is sufficient and ADR-0001 keeps hosted lifecycle out
   of this reusable library.
4. **Select the larger likelihood or lower information criterion.** Rejected as
   insufficient for non-nested preference under ADR-0006.

## Reversal conditions

Supersede this ADR only if a different owner supplies an equally versioned,
content-addressed, Rust-backed, relation-safe artifact with the same or stronger
recovery and interoperability evidence.

## References

American Educational Research Association, American Psychological
Association, & National Council on Measurement in Education. (2014).
*Standards for educational and psychological testing*. American Educational
Research Association.

Schneider, L., Chalmers, R. P., Debelak, R., & Merkle, E. C. (2020). Model
selection of nested and non-nested item response models using Vuong tests.
*Multivariate Behavioral Research, 55*(5), 664–684.
https://doi.org/10.1080/00273171.2019.1664280

Vuong, Q. H. (1989). Likelihood ratio tests for model selection and non-nested
hypotheses. *Econometrica, 57*(2), 307–333.
https://doi.org/10.2307/1912557
