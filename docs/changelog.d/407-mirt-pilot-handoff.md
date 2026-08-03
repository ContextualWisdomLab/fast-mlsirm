# Pilot-observation handoff to binary MIRT calibration

## Added

- A factory-sealed, content-addressed `MirtPilotDesign` assembled by
  `build_mirt_pilot_design` from replay-verified pilot observation records:
  a deterministic persons-by-items binary response matrix with explicit
  `missing`, `not_applicable`, and `insufficient_evidence` states preserved
  alongside the numeric `NaN` representation, per-cell rater assignments
  retained as provenance, and copied `responses`/`factor_id` arguments
  accepted directly by the existing Rust-backed `fast_mlsirm.fit` API.
- Items are assigned to trait dimensions by sorted `query_testlet_id`
  (simple-structure: one dimension per query testlet), and the full mapping
  is disclosed through `factor_testlet_ids`/`item_factor_ids` inside the
  content-addressed design identity.
- The handoff is fail-closed: mixed pilot studies, conflicting item
  provenance, duplicate respondent-item cells (multi-rater data must use
  `build_facets_pilot_design`), non-binary observed categories (polytomous
  data likewise), unobserved respondents or items, and dense designs above
  the documented `MAX_MIRT_PILOT_CELLS` budget are all rejected with
  structured `PilotObservationError` codes before allocation. No silent
  rater aggregation or dichotomization is ever performed, and the handoff
  makes no scoreability, fit, or validity claim. BIFAC2PLM, testlet, DIF,
  and G-theory handoffs remain follow-up slices of issue #407.
