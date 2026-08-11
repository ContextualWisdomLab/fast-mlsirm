# Fit-statistics Rust tail ownership

## Standards

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B (Methodological), 57*(1), 289–300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

## Rationale

Chi-square survival and Benjamini–Hochberg decisions are pure numeric kernels used in DIF and item-fit FDR gates. Owning them in Rust keeps psychometrics math on the multi-threaded CPU path and prevents divergent NumPy re-implementations.

## Implementation

- `mlsirm_core::fitstats::{chi2_sf, benjamini_hochberg}`
- PyO3 exports on `_core`
- Python marshalling in `fast_mlsirm.fitstats`
