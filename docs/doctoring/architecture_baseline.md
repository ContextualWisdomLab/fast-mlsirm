# Architecture baseline — doctoring note

## Claim

The repository documents a layered architecture with a Rust primary numeric core
(CPU multithreaded + optional GPU), Python orchestration, multilevel/multigroup
population structures, and recovery-style verification as the commercial quality
bar.

## APA 7th sources

Fox, J.-P., & Glas, C. A. W. (2001). Bayesian estimation of a multilevel IRT
model. *Psychometrika, 66*(2), 271–288. https://doi.org/10.1007/BF02294839

Jeon, M., Jin, I. H., Schweinberger, M., & Baugh, S. (2021). Mapping unobserved
item-respondent interactions: A latent space item response model with
interaction map. *Psychometrika, 86*(2), 378–403.
https://doi.org/10.1007/s11336-021-09762-5

Kang, I., & Jeon, M. (2025). Multidimensional latent space item response models:
A note on the relativity of conditional dependence. *Psychometrika, 90*(2),
799–826. https://doi.org/10.1017/psy.2025.5

## Traceability

- Code: `crates/mlsirm-core`, `crates/fast-mlsirm-py`, `python/fast_mlsirm`
- Living doc: `/ARCHITECTURE.md`
- Related design: `docs/mmle_marginal_lsirm_design.md`
