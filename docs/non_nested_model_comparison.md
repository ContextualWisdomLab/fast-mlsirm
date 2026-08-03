# Decision-safe non-nested model comparison

`fast_mlsirm.model_comparison.compare_nonnested_models` converts the existing
Rust-backed Vuong statistic into an auditable, fail-closed model-selection
result. The API is intended for paired **casewise marginal log-likelihood**
contributions from two models fitted to the same observations.

## Verified statistic

For case `i`, define

\[
m_i = \ell_i^{(A)} - \ell_i^{(B)},
\qquad
\bar m = \frac{1}{N}\sum_{i=1}^{N}m_i,
\]

and

\[
\widehat{\omega}^2
= \frac{1}{N}\sum_{i=1}^{N}(m_i-\bar m)^2.
\]

For a declared strictly non-nested comparison, the uncorrected statistic is

\[
Z = \frac{\sum_i m_i}{\sqrt{N}\,\widehat{\omega}}.
\]

With `bic_correction=True`, the Rust kernel applies the Schwarz/BIC penalty

\[
Z_{BIC}
= \frac{\sum_i m_i
- \tfrac{1}{2}(k_A-k_B)\log N}
{\sqrt{N}\,\widehat{\omega}}.
\]

Positive values favor model A. The Python module performs no likelihood,
variance, correction, standardization, or probability calculation; those
quantities come from `fast_mlsirm._core` through
`fast_mlsirm.fitstats.vuong_nonnested`.

## Fail-closed decision sequence

1. Supply distinct, auditable model labels and declare the mathematical
   relationship: `strictly_non_nested`, `overlapping`, `nested`,
   `boundary_nested`, or `unknown`.
2. The Rust core computes `mean_diff`, `omega`, `z`, and the two-sided normal
   p value.
3. If the Rust kernel reports exact zero variance, the decision wrapper returns
   `variance_degenerate` instead of leaking the low-level exception. In that
   exceptional state `omega` is reported as zero and
   `raw_mean_loglik_difference` as `NaN`; the wrapper does not reproduce the
   likelihood moment calculation in Python.
4. `omega_tol` is used only as a numerical variance floor. It is **not** called
   Vuong's formal distinguishability hypothesis test.
5. A model preference is returned only when the relation is declared strictly
   non-nested, the variance is numerically non-degenerate, and the two-sided p
   value is below `alpha`.
6. Nested and boundary-nested declarations suppress the normal-theory result
   and return `requires_likelihood_ratio`.
7. Overlapping declarations return `requires_distinguishability_test` because
   the formal weighted-chi-square test needs casewise score vectors and
   information matrices that are not yet exposed consistently across every
   model family.
8. Unknown relationships return `unknown_relation`; the library does not guess
   nestedness from parameter counts.

## Example

```python
from fast_mlsirm.model_comparison import compare_nonnested_models

result = compare_nonnested_models(
    loglik_a=mls2plm_casewise_loglik,
    loglik_b=bifactor_casewise_loglik,
    k_a=mls2plm_parameter_count,
    k_b=bifactor_parameter_count,
    model_a="MLS2PLM",
    model_b="BIFAC2PLM",
    relation="strictly_non_nested",
    bic_correction=True,
    alpha=0.05,
)

print(result.status)
print(result.z, result.p_two_sided)
print(result.preferred_model)
```

The result preserves the available Rust audit fields even when inference is
suppressed:

- `raw_mean_loglik_difference`
- `omega`
- `variance_positive`
- model labels and parameter counts
- declared relation and explicit status
- warning explaining the required alternative procedure

For exact zero variance the strict Rust kernel returns an indistinguishability
error before exposing the raw mean. The decision wrapper therefore reports the
state rather than inventing a Python-side mean.

## Scope boundaries

### Independent observations

The current normal calibration assumes the supplied casewise contributions are
independent sampling units. Repeated claims, judges, prompt variants, or turns
from the same query must not be passed as independent cases without a justified
sampling model. Cluster aggregation and bootstrap resampling are intentionally
absent from this API rather than being implemented ad hoc in Python. A future
cluster-robust extension must be a Rust kernel with a documented asymptotic
contract and recovery study.

### Formal distinguishability

The condition `omega > omega_tol` is a numerical guard only. Vuong's formal
null of observational equivalence uses a weighted chi-square distribution for
`N * omega_hat^2`; the weights depend on model score and information matrices.
Schneider et al. show why this step matters in IRT model selection. Until those
inputs are available for all candidate families, overlapping-model preference
is suppressed.

### Nested and boundary cases

Ordinary nested models require the relevant likelihood-ratio reference
distribution. Parameters on a boundary, such as a variance fixed at zero, can
require a mixture distribution or parametric bootstrap. The API reports this
state instead of applying the non-nested normal reference distribution.

## References

- Vuong, Q. H. (1989). Likelihood ratio tests for model selection and
  non-nested hypotheses. *Econometrica, 57*(2), 307-333.
  https://doi.org/10.2307/1912557
- Schneider, L., Chalmers, R. P., Debelak, R., & Merkle, E. C. (2020). Model
  selection of nested and non-nested item response models using Vuong tests.
  *Multivariate Behavioral Research, 55*(5), 664-684.
  https://doi.org/10.1080/00273171.2019.1664280
- Merkle, E. C., You, D., & Preacher, K. J. (2016). Testing nonnested
  structural equation models. *Psychological Methods, 21*(2), 151-163.
  https://doi.org/10.1037/met0000038
