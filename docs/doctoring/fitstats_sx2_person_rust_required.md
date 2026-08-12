# S-X² and person-fit Rust numerical ownership

## Literature

Orlando, M., & Thissen, D. (2000). Likelihood-based item-fit indices for dichotomous item response theory models. *Applied Psychological Measurement, 24*(1), 50–64. https://doi.org/10.1177/01466216000241003

Drasgow, F., Levine, M. V., & Williams, E. A. (1985). Appropriateness measurement with polychotomous item response models and standardized indices. *British Journal of Mathematical and Statistical Psychology, 38*(1), 67–86. https://doi.org/10.1111/j.2044-8317.1985.tb00817.x

Snijders, T. A. B. (2001). Asymptotic null distribution of person fit statistics with estimated person parameter. *Psychometrika, 66*(3), 331–342. https://doi.org/10.1007/BF02294437

## Product application

Ordinary public S-X² and person-fit results are produced only by the compiled Rust core. Missing or incomplete extension modules raise `RuntimeError` before Python reference grids or NumPy probability arithmetic run. Trait priors for S-X² are marshaled into the native entrypoint rather than reopening a second numerical owner.

## Verification

- `tests/test_fitstats_rust_ownership_failclosed.py`
- `tests/test_fitstats.py` incomplete-core regression
