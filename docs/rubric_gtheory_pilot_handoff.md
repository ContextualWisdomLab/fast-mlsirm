# Generated-item pilot handoff to one-facet G theory

`GTheoryPiPilotDesign` is the governed bridge from replay-verified generated-item
pilot observations to the repository's Rust-backed one-facet generalizability-
theory APIs. It produces a complete balanced persons-by-items (`p x i`) score
matrix for `gtheory_pi` and `phi_lambda` without deleting cases, imputing scores,
aggregating raters, or performing psychometric arithmetic in Python.

## Contract

`build_gtheory_pi_pilot_design(records)` first delegates to
`build_facets_pilot_design`. The existing many-facet assembler remains the source
of truth for:

- one pilot-study identity;
- complete item-generation, rubric, query/testlet, judge-policy, occasion, and
  admitted-pilot provenance;
- explicit `observed`, `missing`, `not_applicable`, and
  `insufficient_evidence` response states;
- bounded ordered response categories;
- duplicate respondent-item-rater rejection;
- observed support for every indexed respondent, item, and rater; and
- bounded persons-by-items-by-raters allocation.

The G-theory handoff then narrows that design to exactly one observed rater, one
declared occasion, at least two respondents, at least two items, and an observed
score in every respondent-item cell. The rater identity, occasion identity, full
nested facets design, and complete item provenance are included in the SHA-256
design fingerprint.

`scores_array()` returns a fresh `float64` persons-by-items matrix.
`to_gtheory_pi_kwargs()` validates a bounded sequence of proposed D-study item
counts and emits arguments accepted directly by `fast_mlsirm.gtheory_pi`.
`to_phi_lambda_kwargs()` adds a finite mastery cut and emits arguments accepted
directly by `fast_mlsirm.phi_lambda`.

## Why this is explicitly a `p x i` handoff

The current pilot provenance binds `occasion_id` to each admitted item. It does
not yet provide a stable cross-occasion item-family identity that proves two
occasion-specific records represent repeated administrations of the same item.
Consequently this handoff does **not** construct a `p x i x o` tensor and never
relabels raters as occasions. Either shortcut would confound facets and produce
a statistically misleading audit artifact.

A future `p x i x o` bridge requires a separately reviewed schema extension that
binds repeated observations to a stable item-family identifier while retaining
the exact occasion-specific admission and response provenance.

## Missingness and balance boundary

The Rust `gtheory_pi` and `phi_lambda` implementations require complete balanced
data. The nested `FacetsPilotDesign` can preserve missing, not-applicable, and
insufficient-evidence cells, but `GTheoryPiPilotDesign` rejects any such cell with
the structured error `gtheory_pi_incomplete_design`. It never performs complete-
case deletion, itemwise deletion, mean substitution, failure-score coercion, or
model-based imputation.

A study with unbalanced data, nested facets, mixed/fixed facets, or missing data
must use an explicitly specified estimator whose assumptions and identification
conditions are independently reviewed.

## Interpretation boundary

A successful handoff establishes only that the governed pilot records can be
reconstructed as the complete balanced input expected by the existing one-facet
G-theory kernel. It does not establish that:

- respondents and items are sampled as random facets from defensible universes;
- the one-facet crossed ANOVA model is appropriate;
- one rater or one occasion is representative of operational conditions;
- the implementation's component-wise negative-variance clamp is the preferred
  estimator for the study;
- a proposed D-study item count is operationally feasible;
- `E-rho^2`, `Phi`, or `Phi(lambda)` satisfies a universal cutoff;
- generated items are invariant, fair, scoreable, or valid; or
- the resulting scores are suitable for consequential decisions.

Buyers should predeclare the universe of admissible observations, facet status,
decision type, D-study design, and missing-data policy; inspect raw and clamped
variance components; run recovery and sensitivity analyses; and combine the
results with DIF, local-dependence, scoreability, fairness, and human-anchored
validity evidence.

## Verified methodological source

Huebner, A., & Lucht, M. (2019). Generalizability theory in R. *Practical
Assessment, Research, and Evaluation, 24*, Article 5.
https://doi.org/10.7275/5065-gc10

The repository's Rust module documents that this article was read in full and
that its worked examples were reproduced. The mean-square-to-variance-component
inversions are hand-derived from the standard expected mean squares and
numerically verified against the article's published tables. Brennan (2001) and
Shavelson and Webb (1991) remain cited there only through the verified article;
no claim is made that their unread derivations were independently reproduced.
