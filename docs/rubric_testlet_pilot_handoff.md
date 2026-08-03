# Generated-item pilot handoff to testlet calibration

`TestletPilotDesign` is the governed bridge from replay-verified binary pilot
observations to the existing Rust-backed `fast_mlsirm.fit_testlet` API. It does
not parse provider output, mutate generated content, aggregate raters, or
perform psychometric arithmetic. Those boundaries remain in the generated-item
validation, audit, pilot-admission, and observation-contract layers.

## Contract

`build_testlet_pilot_design(records)` first delegates to
`build_mirt_pilot_design`. The binary assembler remains the source of truth for:

- one pilot-study identity;
- item provenance and exact schema compatibility;
- one response per respondent-item cell;
- binary observed categories without silent dichotomization;
- explicit `missing`, `not_applicable`, and `insufficient_evidence` states;
- retained per-cell rater provenance;
- observed support for every respondent and item; and
- the bounded dense persons-by-items allocation.

The handoff maps the assembler's disclosed, sorted `query_testlet_id` values to
zero-based integer `testlet_id` values accepted by `fit_testlet`. At least one
query testlet must contain two or more items. A collection containing only
singleton groups is rejected rather than being labelled as a testlet design.
The descriptive mapping and the complete nested binary design are included in
the SHA-256 design fingerprint.

`to_fit_testlet_kwargs` returns fresh response and testlet arrays plus bounded,
validated execution settings. Supported model labels are `rasch` and `2pl`;
quadrature sizes and resource controls match the public testlet estimator.
Rust-core availability and numerical convergence remain estimator concerns.

## Interpretation boundary

The testlet response model represents within-stimulus local dependence through
a person-by-testlet random effect. A successful handoff establishes only that
the pilot observations and testlet grouping can be passed reproducibly to that
model. It does not show that:

- the design is connected or adequately powered;
- the estimator converges;
- testlet variances differ from zero;
- a testlet model fits better than conditional-independence, bifactor, or other
  local-dependence alternatives;
- scores are reliable, fair, or valid; or
- generated items are suitable for operational or high-stakes use.

Those conclusions require exact-head estimation evidence, model comparison,
parameter-recovery studies, residual diagnostics, DIF/fairness analysis, and
human-anchored validity evidence.

## Primary methodological sources

Bradlow, E. T., Wainer, H., & Wang, X. (1999). A Bayesian random effects model
for testlets. *Psychometrika, 64*(2), 153–168.
https://doi.org/10.1007/BF02294533

Wang, X., Bradlow, E. T., & Wainer, H. (2002). A general Bayesian model for
testlets. *Applied Psychological Measurement, 26*(1), 109–128.
https://doi.org/10.1177/0146621602026001007
