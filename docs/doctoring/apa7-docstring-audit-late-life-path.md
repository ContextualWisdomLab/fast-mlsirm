# APA 7th docstring citation audit — late-life reanalysis code path

## Scope

Public functions/structs on the code path used by the late-life reanalysis
(per its `library_migration_map.md`) in:

- `python/fast_mlsirm/polytomous.py`
- `python/fast_mlsirm/bifactor_grm.py`
- `python/fast_mlsirm/dif.py`

and the Rust modules they call in `crates/mlsirm-core/src/`:

- `poly.rs`, `bifactor_grm.rs`, `two_tier_grm.rs`, `quadrature.rs`, `dif.rs`,
  `linking.rs`

Per AGENTS.md's maintainer rule, every docstring/doc comment implementing a
method must state its basis as an APA 7th in-text citation with a verified
page/equation locator, plus an APA 7th reference entry, citing only full
texts actually read.

## Method

Static scan of each public `def`/`class` (Python) and `pub fn` (Rust) on the
scoped files for: presence of a docstring, presence of an author/year
citation token, and presence of a page/equation/section locator (`p.`,
`pp.`, `eq.`, `equation`, `section`, `§`). This is a coverage census, not a
verification pass — locator presence does not confirm the locator was
checked against a full text, only that a docstring gestures at one.

## Disposition

No docstring edits were made in this PR. Fixing the flagged gaps requires
reading each implementation's source paper in full (Zotero / `~/papers` /
KW library, per AGENTS.md) to write a verifiable page/equation locator; that
verification could not be completed for this many symbols in one pass
without risking unverified citations, which the maintainer rule explicitly
forbids ("citing only full texts actually read"). All gaps are instead
filed as a single tracking issue (see PR description) for follow-up,
paper-by-paper.

## Gap census (functions with docstring but no locator marker, or no
citation token at all, are gaps; Rust functions with no doc comment are
also gaps since they implement a method with no stated basis)

### `python/fast_mlsirm/polytomous.py`

Gaps (no author-year token found, or no locator): `PolytomousFit`,
`polytomous_category_probabilities`, `polytomous_expected_response`,
`ExpectedScoreMonotonicity`, `expected_total_score_monotonicity`,
`focal_expected_total_score_monotonicity`,
`bifactor_expected_total_score_monotonicity`, `fit_polytomous`,
`score_polytomous`, `information_polytomous`, `PolyLsirmFit`,
`fit_lsirm_polytomous`, `polytomous_information_criteria`,
`item_fit_polytomous`, `m2_polytomous`, `local_dependence_polytomous`,
`NominalFit`, `person_fit_polytomous`, `cat_simulate_polytomous`,
`dif_polytomous_purified`, `dif_polytomous_anchor_sets` (no docstring at
all), `u3_person_fit_polytomous`, `u3_cutoff_polytomous`.

Has both citation and locator: `fit_nominal_polytomous`, `dif_polytomous`.

### `python/fast_mlsirm/bifactor_grm.py`

Gaps: `BifactorGrmFit`, `fit_bifactor_grm` (citation present, no locator).

### `python/fast_mlsirm/dif.py`

Gaps (no locator, or no docstring): `mantel_haenszel_dif_purified`,
`sibtest` (no docstring — has module text nearby but not a `"""..."""`
docstring), `raju_area`, `logistic_dif` (no docstring).

Has citation and locator: `mantel_haenszel_dif`, `logistic_dif_purified`,
`mantel_smd_dif`, `gmh_dif`, `breslow_day_dif`.

### `crates/mlsirm-core/src/poly.rs` (18 `pub fn`)

Only `u3_poly_person_fit` carries any citation token, and it lacks a
locator. The other 17 (`grm_logprobs`, `grm_node_gradient`,
`gpcm_logprobs`, `compute_gpcm_node_gradient` (formerly `gpcm_node_gradient`), `polytomous_predictions`,
`fit_poly_unidim`, `fit_nominal`, `poly_person_fit`, `poly_cat_next_item`,
`poly_cat_simulate`, `fit_poly_multigroup`, `poly_dif_sweep`,
`u3_poly_bootstrap_cutoff`, `poly_item_information`,
`poly_information_curves`, `score_poly_eap`, `poly_s_x2`) have no citation
token; several also have no `///` doc comment at all.

### `crates/mlsirm-core/src/bifactor_grm.rs` (4 `pub fn`)

All four (`fit_bifactor_grm`, `bifactor_grm_marginal_loglik`,
`bifactor_grm_marginal_loglik_brute`, `fit_bifactor_grm_multigroup`) have no
`///` doc comment at the `pub fn` site (basis is stated elsewhere in the
file/module docs, not verified here).

### `crates/mlsirm-core/src/two_tier_grm.rs` (3 `pub fn`)

Same as above: `fit_two_tier_grm`, `two_tier_grm_marginal_loglik`,
`two_tier_grm_marginal_loglik_brute` have no `///` doc comment at the `pub
fn` site.

### `crates/mlsirm-core/src/quadrature.rs`

No `pub fn` items (module exposes other item kinds); out of this audit's
`pub fn`-based scope.

### `crates/mlsirm-core/src/dif.rs` (12 `pub fn`)

Gaps (no locator, or no doc comment): `as_str`, `mantel_haenszel_dif`,
`logistic_dif`, `mantel_haenszel_dif_purified`, `sibtest`, `raju_area` (no
doc comment), `delta_plot`, `eb_mh_dif`.

Has citation and locator: `logistic_dif_purified`, `mantel_smd_dif`,
`gmh_dif`, `breslow_day_dif`.

### `crates/mlsirm-core/src/linking.rs` (3 `pub fn`)

Gaps: `parse` (no doc comment), `irt_link` (no doc comment),
`link_fixed_item_parameters` (doc comment present, no citation token).

## Totals

- Python (3 files, 34 public symbols): 27 gaps, 7 already citation+locator
  complete.
- Rust (6 files, 40 `pub fn`): 37 gaps, 3 already citation+locator complete.

## Follow-up

Tracked in the GitHub issue opened alongside this PR. Each gap should be
closed by reading the implementing paper in full and adding a verified
page/equation locator plus an APA 7th reference entry, one paper (not one
function) at a time, since most gaps within a file share the same source.
