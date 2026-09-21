## Summary

Contract gate **G3** (`docs/numeric_integration_contract_20260921.json`, commit `bc48fcf`, sha256 `7da18b6d…e464a9`; not yet on `main`) maps the ten Table 3 *p* cells to column `p_wald_two_sided` of `results/library_regression_h1_h5_coefficients.csv`. The driver did not emit that column. This PR adds it.

- **Definition.** Two-sided HC3 Wald test of β_j = 0: `p = fast_mlsirm.chi2_sf_df1((estimate / HC3_SE)^2)` = P(χ²₁ ≥ W) = P(|N(0,1)| ≥ |b/SE|). It uses the SAME `fit_ols_hc` call's estimate and HC3 SE that the row already reports.
- **Not done:** no new fit, no substitute statistic (for example t/F), and no layout-side calculation. The p comes from the fast-mlsirm Rust tail the contract names; the provenance `tails` field already lists `fast_mlsirm.chi2_sf_df1`.
- **CSV schema:** `term, estimate, HC3_SE, ci_lower, ci_upper, p_wald_two_sided` (appended last).

## Test (consumer path)

`tests/test_library_regression_h1_h5_p_wald.py` runs the real driver pipeline (`build_design → fit_ols_hc → all_estimands → write_outputs`) on synthetic, heteroskedastic data with no participant rows. It then reads the written CSV the way the contract consumer does (`term` key → `p_wald_two_sided`) and checks each of the 10 cells:
- `p == chi2_sf_df1((estimate/HC3_SE)^2)` exactly, from the same row;
- a match to an independent two-sided normal tail `erfc(|z|/√2)` (rel 1e-9);
- 0 ≤ p ≤ 1, with the set spanning small and non-significant values.

The embedded 10 (locator, key) cells are checked against the contract file whenever the file is present. It is not on `main` yet, so that check skips there. I ran it locally against a temporary copy of the contract.

Local run (fast-mlsirm 0.11.4 wheel, real Rust core, Python 3.14):
- on `origin/main`'s driver: **1 failed** (`KeyError: 'p_wald_two_sided'`);
- on this branch: **1 passed, 1 skipped** (contract absent), and **2 passed** with the contract copy present.

## Not in scope / open

- **G2 release pin:** `requirements.txt` pins fast-mlsirm `3be36281` (2026-09-12), which predates the regression API (`fit_ols_hc`/`chi2_sf_df1`, added 2026-09-17). The driver already depends on the newer API. Moving the pin to an immutable release is the G2 gate's owner's job; this PR does not change it.
- **G1 scale scores:** not affected. No accepted W/E score artifact means no real p values are produced yet.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_013tDFdELn99o7js5iA4GB1h
