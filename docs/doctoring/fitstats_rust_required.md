# Fit-statistics Rust ownership

## Decision

Public chi-square survival and Benjamini–Hochberg decisions require the compiled Rust core. When `_core_module()` is missing or lacks the entrypoints, the package raises `RuntimeError("fit statistics require the compiled Rust core")` rather than recomputing in pure Python.

## Standards

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B, 57*(1), 289–300.
