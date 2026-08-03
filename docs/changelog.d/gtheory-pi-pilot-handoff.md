# Pilot-observation handoff to one-facet G theory

## Added

- A factory-sealed, content-addressed `GTheoryPiPilotDesign` assembled from the
  existing replay-verified many-facet pilot design. It preserves complete item
  and response provenance while requiring exactly one rater, one declared
  occasion, at least two respondents, at least two items, and an explicitly
  observed score in every respondent-item cell.
- `to_gtheory_pi_kwargs` and `to_phi_lambda_kwargs` emit copied `float64`
  persons-by-items matrices plus bounded D-study item counts and a finite mastery
  cut accepted directly by the repository's Rust-backed one-facet G-theory APIs.
  Missingness, case deletion, imputation, rater aggregation, and score coercion
  fail closed rather than being hidden at the handoff boundary.
- The artifact is explicitly limited to the complete balanced `p x i` design.
  It does not relabel raters as occasions or fabricate a `p x i x o` tensor from
  item-level occasion provenance; a multi-occasion bridge remains deferred until
  the schema can bind repeated administrations to a stable cross-occasion item
  family. No universal coefficient cutoff, variance-component policy, fairness,
  scoreability, or validity claim is made.
