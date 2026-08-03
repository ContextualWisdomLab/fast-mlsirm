# Bifactor scoreability diagnostics

A bifactor model can fit better than a correlated-traits, second-order, testlet,
or latent-space alternative without making every resulting score interpretable.
`mlsirm_core::bifactor_indices` therefore treats **model selection** and
**scoreability** as separate decisions:

1. select or retain a candidate structure using likelihood, predictive,
   recovery, invariance, and appropriate nested/non-nested comparisons;
2. examine whether its general and residual domain factors support defensible
   score reports;
3. report the indices and uncertainty rather than applying an undocumented
   universal cutoff.

The computation is implemented entirely in Rust. No Python or NumPy numerical
fallback is used.

## Inputs

`bifactor_indices` accepts:

- a row-major standardized loading matrix with shape
  `n_items x n_factors`;
- one uniqueness per item;
- the zero-based general-factor column;
- a structural-zero tolerance.

Factors are assumed orthogonal. Every item and factor must have at least one
structurally non-zero loading. Standardized loadings must be finite and have
absolute value below one; uniquenesses must be finite and non-negative.

For fitted logistic-IRT slopes, use the explicitly named
`bifactor_latent_response_indices_from_logit_slopes`. With unit-variance
orthogonal factors and logistic residual variance `pi^2 / 3`, it computes

\[
\lambda_{if}
=
\frac{a_{if}}
{\sqrt{\sum_h a_{ih}^2+\pi^2/3}},
\qquad
\psi_i
=
\frac{\pi^2/3}
{\sum_h a_{ih}^2+\pi^2/3}.
\]

The implementation performs this normalization in scaled coordinates to avoid
intermediate overflow for large finite slopes. This conversion is specific to
the logistic link and must not be applied to probit or other links.

### Logistic latent-response boundary

The slope conversion produces a standardized **continuous latent-response**
loading solution. Consequently, `omega_total` and `omega_hierarchical` returned
by this entry point are continuous latent-response coefficients. They are not
Green-Yang categorical omega coefficients for an observed binary or ordinal sum
score, because category thresholds and observed-score covariance are not inputs
to the function. Operational categorical-score reliability requires a separate
threshold-aware implementation and must not be inferred from these values.

ECV, item ECV, PUC, and `H` likewise describe the transformed loading solution;
they do not by themselves validate an operational score or decision rule. The
long function name is intentional so the scale convention remains visible at
every call site.

## Returned indices

Let `g` denote the general-factor column and let `I_if` indicate that item `i`
has a structural loading on factor `f`.

### Explained common variance

Within the item domain of factor `f`:

\[
\operatorname{ECV}_{SS,f}
=
\frac{\sum_i I_{if}\lambda_{if}^2}
{\sum_i I_{if}\sum_h\lambda_{ih}^2}.
\]

Relative to common variance across the whole bank:

\[
\operatorname{ECV}_{SG,f}
=
\frac{\sum_i I_{if}\lambda_{if}^2}
{\sum_i\sum_h\lambda_{ih}^2}.
\]

General-factor saturation inside each factor's item domain:

\[
\operatorname{ECV}_{GS,f}
=
\frac{\sum_i I_{if}\lambda_{ig}^2}
{\sum_i I_{if}\sum_h\lambda_{ih}^2}.
\]

Item ECV is

\[
I\text{-}ECV_i
=
\frac{\lambda_{ig}^2}
{\sum_h\lambda_{ih}^2}.
\]

### Percentage of uncontaminated correlations

PUC is returned only for a strict bifactor structural pattern: every item loads
on the general factor and on at most one specific factor. If `n_f` items load
on specific factor `f`, then

\[
PUC
=
1-
\frac{\sum_{f\ne g}\binom{n_f}{2}}
{\binom{n_{items}}{2}}.
\]

For cross-loaded, two-tier, or incomplete-general-factor patterns, the API
returns `None`. It does not silently apply a strict-bifactor formula to an
incompatible loading structure.

### Omega total and omega hierarchical

For the item domain associated with factor `f`, let

\[
S_{hf}=\sum_i I_{if}\lambda_{ih}.
\]

Then

\[
\omega_{total,f}
=
\frac{\sum_h S_{hf}^2}
{\sum_h S_{hf}^2+\sum_i I_{if}\psi_i}
\]

and

\[
\omega_{H,f}
=
\frac{S_{ff}^2}
{\sum_h S_{hf}^2+\sum_i I_{if}\psi_i}.
\]

For the general factor, `omega_hierarchical[g]` quantifies variance in the
composite attributable to the general factor. For a specific factor, the same
formula describes the residual target-factor contribution within its item
domain after all orthogonal common factors are represented in the denominator.

### Construct replicability

For each factor:

\[
H_f
=
\frac{1}
{1+\left(\sum_i
\frac{\lambda_{if}^2}{1-\lambda_{if}^2}
\right)^{-1}}.
\]

`H` describes how well a factor is represented by its indicators under the
supplied standardized loading solution. It is not a substitute for external
validity, invariance, or predictive evidence.

## Structural-zero policy

`zero_tolerance` controls the `I_if` membership indicators. Loadings treated as
active retain their original numerical values rather than being rounded to a
threshold value; values below the threshold can still contribute where a
formula sums the full loading row rather than the target-factor membership
term. This prevents a tiny estimation artifact from turning a strict bifactor
pattern into a cross-loaded pattern without silently rewriting the supplied
loading matrix.

The default constructor uses exact zeroes:

```rust
use mlsirm_core::bifactor_indices::{
    bifactor_latent_response_indices_from_logit_slopes,
    BifactorIndicesConfig,
};

let config = BifactorIndicesConfig::new(n_items, n_factors, general_factor);
let diagnostics =
    bifactor_latent_response_indices_from_logit_slopes(&logit_slopes, config)?;
```

Use a non-zero tolerance only when the estimation and sparsity procedure has an
explicit numerical-zero contract.

## Decision boundaries

The kernel intentionally does **not** contain hard-coded pass/fail thresholds.
A defensible deployment policy should also consider:

- uncertainty or bootstrap stability of every index;
- loading and factor recovery under the intended calibration design;
- leave-query, leave-domain, and leave-system predictive performance;
- DIF, judge-family drift, and language/domain invariance;
- local dependence and testlet effects;
- whether the bifactor structure wins an appropriate nested or non-nested
  comparison against simpler alternatives;
- consequential validity of the score report and decision rule.

A high general-factor index does not by itself validate a single total score,
and a fitted specific factor does not by itself justify a subscale.

## Verification oracle

The integration tests reproduce the 12-item, four-factor example distributed by
`BifactorIndicesCalculator` and independently calculate:

- ECV-SS, ECV-SG, and ECV-GS;
- all item ECV values;
- PUC;
- omega total and omega hierarchical;
- construct replicability `H`.

Additional tests cover non-first general-factor placement, single-item specific
factors, cross-loadings, missing general loadings, structural-zero tolerance,
malformed matrices, non-finite inputs, numerical underflow, sign-cancelled
zero-variance composites, and logistic latent-response standardization.

## Source governance and references

The continuous-indicator functions in the following CRAN package source files
were read as implementation oracles:

- `R/ECV_Indices.R`
- `R/Omega_Indices.R`
- `R/Other_Indices.R`

The journal article below is cited as the methodological origin identified by
that package; it was not read in full for this implementation.

Dueber, D. M. (2021). *BifactorIndicesCalculator: Bifactor indices calculator*
(Version 0.2.2) [R package].
https://CRAN.R-project.org/package=BifactorIndicesCalculator

Rodriguez, A., Reise, S. P., & Haviland, M. G. (2016). Evaluating bifactor
models: Calculating and interpreting statistical indices. *Psychological
Methods, 21*(2), 137–150. https://doi.org/10.1037/met0000045
