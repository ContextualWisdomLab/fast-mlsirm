# Evidence: multiple-group bifactor expected raw scores

Scope: the library path that consumes a saved `BifactorMultigroupFit` for
expected raw scores and monotonicity checks **without refitting**.

## The failure this addresses

A consumer's concurrent multiple-group bifactor GRM fit converged, and then
scoring raised:

```
File ".../fast_mlsirm/polytomous.py", in check_bifactor_expected_total_score_monotonicity
    raise ValueError("fit.a_general must be a non-empty 1-D array")
```

The fit was fine; the consume path was not. `BifactorMultigroupFit.a_general`
is `n_groups x n_items`, and the expected-score path accepted only the
single-group 1-D layout.

## What the library now guarantees

Each item says whether a test in
`tests/test_bifactor_multigroup_expected_raw.py` exercises it on synthetic
fits, or whether it holds by construction in `python/fast_mlsirm/polytomous.py`
without a separate test.

- **Group selection is explicit.** `predict_bifactor_expected_total_score(fit,
  theta, q_specific, group=g)` uses group `g`'s `a_general`, `a_specific` and
  `threshold` rows. *Tested* for rows that differ in `a_general`
  (`test_group_specific_rows_require_naming_the_group`: `group=0` and
  `group=1` give different curves, and `group=0` matches the single-group
  curve) and for out-of-range or non-integral `group` values
  (`test_rejects_out_of_range_group`). That `group=g` also picks group `g`'s
  `a_specific` and `threshold` rows holds by construction
  (`_bifactor_group_item_params` indexes all three blocks by the same `g`,
  `python/fast_mlsirm/polytomous.py` lines 632-634);
  no test varies those rows alone.
- **Implicit selection only when it is exact.** `group=None` is accepted for a
  multiple-group fit only if the item-parameter blocks are exactly equal across
  groups, as an all-anchored fit (`anchor_mask=None`) makes them. A fit whose
  rows differ raises rather than silently scoring group 0. *Tested* for
  differing `a_general` rows (`test_group_specific_rows_require_naming_the_group`)
  and differing `threshold` rows (`test_threshold_rows_are_part_of_the_identity_check`).
  The same exact-equality check also covers `a_specific`
  (`_bifactor_group_item_params`, which compares all three blocks at
  `python/fast_mlsirm/polytomous.py` lines 604-609), but
  `a_specific`-only differences have no separate test.
- **Group population parameters stay out of the conditional curve (by
  construction; not separately tested).** `_bifactor_group_item_params`
  returns only one group's `(a_general, a_specific, threshold)` rows
  (`python/fast_mlsirm/polytomous.py` lines 619-642), and `general_mean` /
  `general_sd` are never read on this path (lines 684-715 use only those rows
  and `theta`). Multiple-group item
  parameters and `theta_g_eap` share the one common (reference) metric that the
  E-step places every group's nodes on, so no per-group rescaling of `theta` is
  applied. A test that varies `general_mean`/`general_sd` and checks that the
  curve is unchanged would make this an executable guarantee.
- **Unit specific variances are required (tested: `test_rejects_estimated_specific_sd`).** With `estimate_specific_vars=False`
  the per-item specific-factor marginalization is computable from the fit
  alone. The check is scoped to the group being scored
  (`python/fast_mlsirm/polytomous.py` lines 625-630): with an explicit
  `group=g` only group `g`'s `specific_sd` must be all 1, so `group=0` still
  scores when another group has a non-unit SD; with `group=None` every group's
  `specific_sd` must be all 1. A non-unit SD in the checked scope raises,
  because the fit does not carry the `specific_map` needed to match an item to
  its specific factor.
- **One kernel (tested: `test_monotonicity_check_reads_the_same_kernel`, `test_pointwise_theta_may_tie_and_be_unordered`).** `check_bifactor_expected_total_score_monotonicity` and
  `predict_bifactor_expected_total_score` read the same kernel. Pointwise
  scoring of unordered or tied `theta` agrees with scoring on the sorted unique
  grid.

## The compiled core is on this path

*By construction; not separately tested.* In
`python/fast_mlsirm/polytomous.py`, `predict_bifactor_expected_total_score`
(line 645) calls `predict_expected_response_polytomous` per item (line 712),
which returns `_polytomous_predictions(...)[1]` (line 262); lines 220-221 are
the raise and the native call quoted below.

`predict_bifactor_expected_total_score` calls
`predict_expected_response_polytomous` per item. That routes through
`_polytomous_predictions`, which requires the native extension: it raises
`RuntimeError("polytomous predictions require the compiled Rust core")` when the
module is absent and otherwise calls `core.polytomous_predictions`. The new
group-selection and validation logic is NumPy, but every expected-score value
comes from native code.

## Consumer-side verification

Verification against a consumer's preserved (non-public) fit checkpoint belongs
in the consumer's own repository and is not recorded here. That includes the
checkpoint digest, the loaded native artifact, row/group alignment and any
score summaries. It does not substitute for release acceptance. Acceptance is
owed once an immutable release exists, and it covers release provenance, the
installed core path and hash, the saved fit consumed with no refit, and
index-wise person/row/group alignment.
