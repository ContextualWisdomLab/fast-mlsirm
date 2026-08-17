# Fit-statistics require compiled Rust core

## Fixed

- Public `chi2_sf` and `benjamini_hochberg` fail closed with a stable RuntimeError when the compiled Rust core is unavailable, preventing silent pure-Python numerical ownership.
