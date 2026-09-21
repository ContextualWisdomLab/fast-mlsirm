## Summary

This moves the fast-mlsirm pin from `3be36281` (2026-09-12, before v0.10.0) to **`e44b52d851596910dc7a8fa3f174f2aff03872fa` = tag `v0.11.4`**, which is also the GitHub Release "v0.11.4" (Latest, 2026-09-18) and **PyPI `fast-mlsirm==0.11.4`**. The install route is unchanged (`git+https` source build); only the commit changes. The README pin sentence is updated to match.

**Why:** `analysis/library_regression_h1_h5.py` imports `fit_ols_hc`, `contrast`, `chi2_sf_df1` and `f_sf` (fast-mlsirm #1982, merge `48fde1c3`), and the old pin has none of them. The tag checks: `v0.10.0` does not contain `48fde1c3`; `v0.11.0` through `v0.11.4` all do.

## Provenance proof (release = source)

- PyPI 0.11.4 files were downloaded and **sha256-verified** against the PyPI JSON digests: sdist `761f1378…`, and the cp312 macOS universal2 wheel `d026033b…`.
- **sdist vs tag commit:** all 307 non-`PKG-INFO` files are byte-identical to `git archive e44b52d8` (0 differing, 0 extra).
- **wheel vs tag commit:** the Python sources in `fast_mlsirm/` are identical to the tag. The only differences are the compiled `_core` extension and files that exist only in the wheel.
- **Isolated install:** fresh uv venv, Python 3.12, `fast-mlsirm==0.11.4` from the PyPI index (`direct_url.json` absent, not a `file://` or dev wheel). The installed `_core.cpython-312-darwin.so` sha256 `882e189d…` equals the hash-verified wheel's.
- **At `v0.11.4`:** `__init__` exports `fit_ols_hc`, `contrast`, `chi2_sf_df1` and `f_sf`, and they are defined in `regression.py`.

## Consumer checks (isolated PyPI-0.11.4 venv)

- Every `fast_mlsirm` import or attribute referenced by this repo's Python code (33 names across 24 distinct usages) resolves in 0.11.4: **0 missing**.
- `pytest tests/` on this branch (`PYTHONPATH=analysis`, as `test_quadrature_input_contract` needs): **7 passed, 2 skipped**. The 2 skips are `test_mc_stop_library_contract`, blocked on fast-mlsirm#2013 (MC-stop API, unreleased). That is expected and **not** part of this pin.
- #246's consumer test (`test_library_regression_h1_h5_p_wald.py`) in the same venv: **1 passed, 1 skipped** (contract file absent).
- R tests were not run. Nothing here changes R code.

## Explicitly not in scope

- Unreleased W/E expected-raw APIs (fast-mlsirm #2091 / #2062) and the MC-stop API (#2013). This pin does not provide them.
- Numerical acceptance of any score or table value (gates G1 and G7). This is a dependency pin only.
- Independent of #246. Either can merge first.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_013tDFdELn99o7js5iA4GB1h
