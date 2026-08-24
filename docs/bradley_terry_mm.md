# Bradley–Terry MM pairwise ranking

`fast_mlsirm.bradley_terry_mm` / `mlsirm_core::scaling::bradley_terry_mm`
fits Bradley–Terry worths from an `n × n` paired-comparison win matrix by
Hunter's minorization–maximization (MM) algorithm. The additive-ties variant
`fast_mlsirm.bratt_mm` / `mlsirm_core::scaling::bratt_mm` estimates the same
worths plus a single positive tie parameter `alpha0`.

These estimators are psychometric ranking models. They are not security
controls, and CWE/OWASP/NIST publications are not their methodological basis.

Architecture decision: [`adr/0017-bradley-terry-mm.md`](adr/0017-bradley-terry-mm.md).

## What is implemented

### Tie-free Bradley–Terry (`bradley_terry_mm`)

Model (Bradley & Terry, 1952):

```text
P(i beats j) = w_i / (w_i + w_j)
```

`wins[i, j]` is the (possibly fractional) count of comparisons in which object
`i` beat object `j`. The diagonal must be zero. The MM update (Hunter, 2004)
produces centered log-worths (`params`, mean exactly 0) and exp-scale worths
rescaled to sum `n` (`weights`). Optional Dirichlet-style `alpha` regularization
is added to both the numerator and the denominator of each update.

An all-zero win matrix is rejected for every `alpha`. A contestant with zero
wins at `alpha = 0` has no finite log-worth and is rejected. Non-convergence
within `max_iter` is an error (for example when the comparison graph is not
strongly connected).

### Additive-ties BRATT (`bratt_mm`)

When ties are observed, the implemented model is

```text
P(i beats j) = alpha_i / (alpha_i + alpha_j + alpha0)
P(i ties j)  = alpha0 / (alpha_i + alpha_j + alpha0)
```

`alpha` and `alpha0` are jointly rescaled so that `alpha[ref_index] ==
ref_value`. Tie-free data are rejected (`use bradley_terry_mm`); a contestant
with zero wins is rejected. This is the additive-`alpha0` ties model.

The repository does **not** implement the Rao–Kupper or Davidson ties models.
Those names appear in source comments only to disambiguate the implemented
likelihood.

Python validates arrays and marshals results. Numeric work is in the Rust
kernel.

## Related ranking estimators

Luce Spectral Ranking / I-LSR, Rank Centrality, and Plackett–Luce ranking
kernels are separate estimators. Input-bound doctoring for LSR lives in
[`doctoring/lsr_ranking_input_bounds.md`](doctoring/lsr_ranking_input_bounds.md).
Do not treat those kernels as Bradley–Terry MM, and do not treat Bradley–Terry
MM as a Plackett–Luce ranking likelihood.

## Interpretation boundary

Estimated worths are a paired-comparison scale under the stated model. They do
not by themselves establish:

- a latent IRT trait or MLSIRM coordinate;
- fairness, DIF, or invariance;
- rater interchangeability or many-facet severity;
- a causal ranking of people, models, or products; or
- suitability for high-stakes selection.

Score-use and fairness interpretation remain governed by AERA, APA, and NCME
(2014) when the worths are used as educational or psychological scores.

## Claims not made

- Rao–Kupper (1967) and Davidson (1970) ties models are not implemented.
- Ford (1957) strong-connectivity is not a pre-check; disconnected or
  unbeatable objects fail at estimation time.
- Computational ports of `choix` or VGAM `bratt()` are numerical comparison
  sources, not substitutes for Bradley and Terry (1952) or Hunter (2004).
- Changelog notes that a paper was unread at implementation time are
  historical source-governance comments, not a reason to omit the method's
  bibliographic basis.

## Public Python API

```python
import numpy as np
from fast_mlsirm import bradley_terry_mm, bratt_mm

wins = np.array(
    [
        [0.0, 3.0, 1.0],
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
bt = bradley_terry_mm(wins)
print(bt.params)
print(bt.weights)

ties = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
bratt = bratt_mm(wins, ties, ref_index=0, ref_value=1.0)
print(bratt.alpha)
print(bratt.alpha0)
```

## Primary sources

Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete block
designs: I. The method of paired comparisons. *Biometrika, 39*(3/4), 324–345.
https://doi.org/10.2307/2334029

Hunter, D. R. (2004). MM algorithms for generalized Bradley–Terry models.
*The Annals of Statistics, 32*(1), 384–406.
https://doi.org/10.1214/aos/1079120141

American Educational Research Association, American Psychological
Association, & National Council on Measurement in Education. (2014).
*Standards for educational and psychological testing*. American Educational
Research Association.
