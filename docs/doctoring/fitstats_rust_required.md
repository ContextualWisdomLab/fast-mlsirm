# Fit-statistics Rust ownership

## Decision

Public chi-square survival and Benjamini–Hochberg decisions require the compiled Rust core. When `_core_module()` is missing or lacks the entrypoints, the package raises `RuntimeError("fit statistics require the compiled Rust core")` rather than recomputing in pure Python.

## Standards

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B, 57*(1), 289–300.

## S-X² and person fit

Public `s_x2()` and `person_fit()` require the compiled Rust entrypoints
`s_x2_stat` and `person_fit_stat`. Missing or incomplete cores raise
`RuntimeError("fit statistics require the compiled Rust core")`.

Trait prior shifts supplied via `prior_mean` are transported to the native
S-X² path; ordinary public execution never selects the retained NumPy
reference helpers (`_s_x2_python_reference`, `_person_fit_python_reference`).

## Standards (person fit / item fit)

Orlando, M., & Thissen, D. (2000). Likelihood-based item-fit indices for
dichotomous item response theory models. *Applied Psychological Measurement,
24*(1), 50–64. https://doi.org/10.1177/01466216000241003

Drasgow, F., Levine, M. V., & Williams, E. A. (1985). Appropriateness
measurement with polychotomous item response models and standardized indices.
*British Journal of Mathematical and Statistical Psychology, 38*(1), 67–86.
https://doi.org/10.1111/j.2044-8317.1985.tb00817.x

Snijders, T. A. B. (2001). Asymptotic null distribution of person fit
statistics with estimated person parameter. *Psychometrika, 66*(3), 331–342.
https://doi.org/10.1007/BF02294437

