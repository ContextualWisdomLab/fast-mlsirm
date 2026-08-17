# Fit-statistics tail ownership

## Changed

- Public `chi2_sf` and `benjamini_hochberg` now prefer the Rust core for ranking and tail arithmetic, with a pure-Python fallback only when the compiled core methods are unavailable.
