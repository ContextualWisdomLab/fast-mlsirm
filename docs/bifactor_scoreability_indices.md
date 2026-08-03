# Bifactor scoreability diagnostics

A bifactor model can fit better than correlated-traits, second-order, testlet,
or latent-space alternatives without making every resulting score
interpretable. `fast-mlsirm` therefore keeps two decisions separate:

1. select a defensible candidate model using likelihood, predictive,
   recovery, invariance, and relation-appropriate model comparison evidence;
2. examine whether the retained loading solution supports a general score and
   residual factor-specific scores.

All scoreability arithmetic is implemented in Rust. Python validates array
shape, invokes the compiled kernel, and returns an immutable typed result.
There is no NumPy formula fallback for this post-fit diagnostic.

## Public Python API

```python
import numpy as np
from fast_mlsirm import (
    bifactor_scoreability,
    bifactor_scoreability_from_logit_slopes,
)

loadings = np.asarray(
    [
        [0.70, 0.40, 0.00],
        [0.70, 0.30, 0.00],
        [0.70, 0.00, 0.50],
        [0.70, 0.00, 0.60],
    ],
    dtype=np.float64,
)
uniquenesses = 1.0 - np.square(loadings).sum(axis=1)

result = bifactor_scoreability(
    loadings,
    uniquenesses,
    general_factor=0,
)

print(result.ecv_sg)
print(result.omega_hierarchical)
print(result.construct_replicability)
```

For fitted orthogonal logistic-IRT slopes, use
`bifactor_scoreability_from_logit_slopes`. The Rust core applies the logistic
latent-response residual variance `pi^2 / 3` in overflow-resistant scaled
coordinates.

## Input contract

The standardized-loading entry point requires:

- a finite `n_items x n_factors` loading matrix;
- at least two items and two factors;
- one finite uniqueness in `[0, 1]` per item;
- an in-range zero-based `general_factor` column;
- an optional finite non-negative `zero_tolerance`;
- every item to have an active loading on the declared general factor;
- every factor to have at least one active item loading; and
- the itemwise standardized identity

$$
\sum_f \lambda_{if}^{2}+\psi_i=1
$$

within absolute tolerance `1e-8`.

A loading is structurally active when
`abs(loading) > zero_tolerance`. The threshold classifies the structural
pattern only; numerical formulas retain the supplied loading values.

An incomplete declared general-factor column is rejected. The library does not
emit bifactor-labelled ECV or item-ECV values for a matrix that is not a
bifactor solution under its declared general factor.

## Bounded CPU execution

This is a small post-fit diagnostic rather than an iterative estimator. In the
most general cross-loaded pattern, its work is bounded by

$$
O(n_{items}n_{factors}^{2}).
$$

The Rust and Python boundaries both reject requests above
`50,000,000` `items * factors^2` work units. The independent limits remain
1,000,000 items and 64 factors. Work-budget validation occurs before Rust
input-length checks and before extension dispatch in the Python wrapper.

The accepted operation is a deterministic CPU reduction over a loading matrix.
At this scale, GPU buffer transfer, dispatch, and synchronization would dominate
the arithmetic and add a second numerical execution path without buyer value.
GPU and CPU-multithread acceleration remain mandatory for the estimation layers
where repeated likelihood and gradient work amortizes those costs; this bounded
scoreability kernel intentionally stays in Rust on the CPU.

## Logistic latent-response conversion

For item `i`, orthogonal logistic slopes `a_if`, and residual variance
`pi^2 / 3`, the compiled conversion is

$$
\lambda_{if}
=
\frac{a_{if}}
{\sqrt{\sum_h a_{ih}^{2}+\pi^{2}/3}},
\qquad
\psi_i
=
\frac{\pi^{2}/3}
{\sum_h a_{ih}^{2}+\pi^{2}/3}.
$$

The returned omega values describe the standardized **continuous
latent-response representation**. They are not categorical observed-score
omega coefficients for binary or ordinal sum scores. Threshold-aware
observed-score reliability requires a separate model and must not be inferred
from this output.

## Returned result

`BifactorScoreabilityResult` contains:

- `factor_item_counts`
- `is_strict_bifactor`
- `puc`
- `ecv_ss`
- `ecv_sg`
- `ecv_gs`
- `item_ecv`
- `omega_total`
- `omega_hierarchical`
- `construct_replicability`

The NumPy vectors are read-only. Cross-loaded specific-factor patterns are
valid descriptive loading solutions but are not strict bifactor patterns;
`puc` is therefore `None`. Missing general-factor loadings are errors rather
than a non-strict PUC case.

## Explained common variance

Let `g` be the declared general factor and `I_if` indicate that item `i` is
structurally active on factor `f`.

$$
ECV_{SS,f}
=
\frac{\sum_i I_{if}\lambda_{if}^{2}}
{\sum_i I_{if}\sum_h\lambda_{ih}^{2}},
$$

$$
ECV_{SG,f}
=
\frac{\sum_i I_{if}\lambda_{if}^{2}}
{\sum_i\sum_h\lambda_{ih}^{2}},
$$

$$
ECV_{GS,f}
=
\frac{\sum_i I_{if}\lambda_{ig}^{2}}
{\sum_i I_{if}\sum_h\lambda_{ih}^{2}},
$$

and

$$
I\text{-}ECV_i
=
\frac{\lambda_{ig}^{2}}
{\sum_h\lambda_{ih}^{2}}.
$$

## Percentage of uncontaminated correlations

PUC is defined only when every item has the declared general loading and at
most one active specific-factor loading. If `n_f` items load on specific factor
`f`,

$$
PUC
=
1-
\frac{\sum_{f\ne g}\binom{n_f}{2}}
{\binom{n_{items}}{2}}.
$$

## Omega and construct replicability

For the item domain associated with factor `f`, let

$$
S_{hf}=\sum_i I_{if}\lambda_{ih}.
$$

Then

$$
\omega_{total,f}
=
\frac{\sum_h S_{hf}^{2}}
{\sum_h S_{hf}^{2}+\sum_i I_{if}\psi_i},
$$

$$
\omega_{H,f}
=
\frac{S_{ff}^{2}}
{\sum_h S_{hf}^{2}+\sum_i I_{if}\psi_i},
$$

and

$$
H_f
=
\frac{1}
{1+\left(\sum_i
\frac{\lambda_{if}^{2}}{1-\lambda_{if}^{2}}
\right)^{-1}}.
$$

The kernel does not hard-code universal interpretation cutoffs. A deployment
policy must also consider uncertainty, parameter recovery, predictive
performance, invariance, DIF, local dependence, testlet effects, and the
consequences of the intended score use.

## Verification

The Rust integration suite pins a 12-item, four-factor numerical oracle and
covers:

- ECV-SS, ECV-SG, ECV-GS, and item ECV;
- PUC, omega total, omega hierarchical, and `H`;
- a non-first general-factor column;
- single-item specific-factor domains;
- cross-loaded specific factors;
- rejection of an incomplete general-factor column;
- standardized-identity roundoff and material violations;
- uniqueness bounds, malformed dimensions, non-finite values, underflow,
  overflow, and sign-cancelled zero-variance composites;
- the deterministic CPU work budget; and
- logistic latent-response conversion.

Python tests compare every typed result field directly with the secondary
PyO3 `_bifactor_core` module, verify package-root exports and immutable result
vectors, and enforce the same pre-dispatch work budget. They do not use an
independent Python implementation of the formulas.

## Source governance

The implemented continuous-indicator formulas were independently transcribed
from the complete CRAN `BifactorIndicesCalculator` 0.2.2 source files
`R/ECV_Indices.R`, `R/Omega_Indices.R`, and `R/Other_Indices.R`. That package is
used as the executable numerical oracle.

The complete author-posted text of Rodriguez, Reise, and Haviland (2016a), its
publisher correction, and the bibliographic record and abstract for Rodriguez,
Reise, and Haviland (2016b) were reviewed. Corrected Equations 1, 4, and 7 and
article Equations 3 and 6 are mapped to the Rust contracts in
`docs/papers/bifactor-scoreability-primary-source-verification.md`. The 2016b
article is cited only at the verified bibliographic/abstract level; this record
does not claim that its full text was reviewed.

Dueber, D. M. (2021). *BifactorIndicesCalculator: Bifactor indices calculator*
(Version 0.2.2) [R package].
https://CRAN.R-project.org/package=BifactorIndicesCalculator

Rodriguez, A., Reise, S. P., & Haviland, M. G. (2016a). Applying bifactor
statistical indices in the evaluation of psychological measures. *Journal of
Personality Assessment, 98*(3), 223–237.
https://doi.org/10.1080/00223891.2015.1089249

Rodriguez, A., Reise, S. P., & Haviland, M. G. (2016b). Evaluating bifactor
models: Calculating and interpreting statistical indices. *Psychological
Methods, 21*(2), 137–150. https://doi.org/10.1037/met0000045

Taylor & Francis. (2016). Correction to: Applying bifactor statistical indices
in the evaluation of psychological measures. *Journal of Personality
Assessment, 98*(4), 444. https://doi.org/10.1080/00223891.2015.1117928
