# Two-tier Oakes identification transport

The fit estimator records `primary_identification`: `orthogonal` for a fixed
identity primary covariance, `correlated` for estimated primary correlations.
`two_tier_oakes_se_from_fit` consumes this record and forwards `identity` or
`estimate` to the existing array/native SE routine. It never infers estimation
history from the numerical Phi matrix. A correlated fit with Phi equal to I
still selects estimated mode. Missing/unknown records, mode overrides, and an
orthogonal record with nonidentity Phi are refused.

The returned `TwoTierOakesSe` carries `primary_correlation` and
`identification_source` (`fit.primary_identification` or `explicit_array_mode`).
This is transport provenance, not a new validity or convergence judgment.
The fit object is mutable: these fields do not authenticate arbitrary caller
edits or establish that supplied responses/maps are the original fit inputs.
They do not make historical array-only checkpoints self-authenticating.

## Compatibility

- `fit_two_tier_grm` retains its existing `estimate` default and numerical code.
- Both Python and native `two_tier_oakes_se` array APIs now require explicit
  `primary_correlation`. Previously omitted calls must supply the known fitting
  mode or use the fit-record adapter. Do not fill missing records by testing Phi.
- Existing explicit positional/keyword array calls keep their argument order.
- The package-root export and direct module export include the new adapter.
- `TwoTierOakesSe` manual construction requires the two new provenance fields.
- `tests/test_two_tier_grm.py` keeps its fit-default comparison but makes the SE
  estimate-mode call explicit; `tests/test_two_tier_oakes.py` does the same for
  the array smoke fixture. Repository Rust callers already use an explicit
  `TwoTierOakesConfig`; their behavior is unchanged.
- `io.save_fit_result` serializes a different `FitResult` model. No two-tier
  checkpoint format is added. Consumers retaining `TwoTierGrmFit` must preserve
  its existing required identification field; older missing records are HOLD.

## Verification boundary

The new Python tests mock only the native computation and check actual adapter
forwarding, failure before native calls, source labeling, and public/native
signature source contracts. They do not establish numerical SE accuracy or
compile/load the changed PyO3 signature. Native verification and independent
review remain required before this candidate is accepted.
