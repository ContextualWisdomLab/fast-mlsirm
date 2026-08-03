# Pilot-observation handoff to many-facet calibration

## Added

- Factory-built, content-addressed `PilotObservationRecord` values that bind
  respondent and rater responses to the complete replay-verified pilot item
  provenance.
- Explicit `observed`, `missing`, `not_applicable`, and
  `insufficient_evidence` states; non-observed states cannot carry a category
  and are never coerced to failure scores.
- A deterministic `FacetsPilotDesign` assembler that rejects mixed studies,
  conflicting item provenance, duplicate cells, invalid category regimes, and
  unobserved indexed facets before producing copied arguments for the existing
  Rust-backed `fit_facets` API.
- Content-addressed preservation of the exact response-state tensor alongside
  the numeric `NaN` representation used for many-facet estimation.
- This handoff performs no psychometric arithmetic and makes no adequacy,
  connectedness, fairness, scoreability, calibration, or validity claim.
  MIRT, bifactor, testlet, DIF, and G-theory handoffs remain follow-up slices
  of issue #407.
