# Pilot-observation handoff to many-facet calibration

## Product boundary

The generated-item pipeline now admits candidates into a pilot through a
replay-verified audit. Calibration still needs response data that cannot lose
item provenance or silently convert missingness states into failures. This
slice adds a deterministic handoff to the existing Rust-backed
`fast_mlsirm.fit_facets` API.

```text
verified PilotCandidateRecord
  -> PilotObservationRecord
  -> FacetsPilotDesign
  -> fit_facets(**design.to_fit_facets_kwargs())
```

## Observation contract

`build_pilot_observation_record` accepts only the public, factory-sealed
`PilotCandidateRecord`. Every observation repeats the pilot-study,
query/testlet, generator-family, judge-policy, occasion, item, and complete
pilot-record fingerprint before adding respondent, rater, response-state, and
category fields.

The response state is one of:

- `observed`: an integer category is required;
- `missing`: no score was obtained;
- `not_applicable`: the response was outside the applicable scoring regime;
- `insufficient_evidence`: the judge explicitly lacked enough evidence.

The three non-observed states require `category=None`. They are never coerced
to category zero or to an incorrect response.

## Deterministic facets design

`build_facets_pilot_design` performs the following fail-closed checks before
creating a persons-by-items-by-raters tensor:

- bounded, typed input records from one pilot study;
- one immutable pilot-provenance binding per item identifier;
- no duplicate respondent-item-rater cell;
- declared or inferable ordered category count within package limits;
- at least one observed response for every indexed respondent, item, and
  rater, matching the requirements of `fit_facets`;
- no more than 1,000,000 cells in the complete dense
  persons-by-items-by-raters cross-product, checked before allocation so a
  sparse record collection cannot amplify into an unbounded tensor;
- deterministic lexicographic index order independent of record order.

Observed-support validation uses precomputed respondent, item, and rater sets,
so it remains linear in supplied records and indexed facets rather than
rescanning all observations for every facet.

The numeric tensor maps every non-observed state to `NaN`, as required by the
existing many-facet fitting API, while the content-addressed design retains the
exact state tensor. The design also retains per-item query/testlet, generator,
judge-policy, occasion, and pilot-record provenance.

## Scientific boundary

This module performs no psychometric arithmetic and does not claim that the
pilot design is adequate, connected, fair, calibrated, or valid. The Rust
many-facet model remains responsible for estimation and reports design
connectedness. Sample-size planning, semantic response adjudication, DIF,
MIRT/bifactor/testlet conversion, G-theory designs, recovery studies, and
high-stakes deployment gates remain separate reviewable slices.
