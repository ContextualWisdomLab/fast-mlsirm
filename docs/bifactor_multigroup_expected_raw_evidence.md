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

## What the library now guarantees (tested)

The guarantees below are exercised by
`tests/test_bifactor_multigroup_expected_raw.py` on synthetic fits:

- **Group selection is explicit.** `predict_bifactor_expected_total_score(fit,
  theta, q_specific, group=g)` uses group `g`'s `a_general`, `a_specific` and
  `threshold` rows. Out-of-range or non-integral `group` values raise.
- **Implicit selection only when it is exact.** `group=None` is accepted for a
  multiple-group fit only if every item-parameter block (`a_general`,
  `a_specific`, `threshold`) is exactly equal across groups, as an all-anchored
  fit (`anchor_mask=None`) makes them. A fit whose rows differ raises rather
  than silently scoring group 0.
- **Group population parameters stay out of the conditional curve.**
  `general_mean` / `general_sd` differ across groups and do not enter
  `E[T | theta_G]`. Multiple-group item parameters and `theta_g_eap` are on the
  one common (reference) metric that the E-step places every group's nodes on,
  so no per-group rescaling of `theta` is applied.
- **Unit specific variances are required.** With `estimate_specific_vars=False`
  the per-item specific-factor marginalization is computable from the fit
  alone. A non-unit `specific_sd` raises, because the fit does not carry the
  `specific_map` needed to match an item to its specific factor.
- **One kernel.** `check_bifactor_expected_total_score_monotonicity` and
  `predict_bifactor_expected_total_score` read the same kernel. Pointwise
  scoring of unordered or tied `theta` agrees with scoring on the sorted unique
  grid.

## The compiled core is on this path

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
