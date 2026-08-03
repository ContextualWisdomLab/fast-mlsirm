# Fail-closed Vuong selection-statistic summary

`fast_mlsirm.model_comparison.compare_nonnested_models` provides an auditable,
resource-bounded wrapper around the existing Rust-backed Vuong **selection
statistic**. It does not declare a winning model before the mathematically
required first-stage distinguishability test has been supplied.

The API accepts paired **casewise marginal log-likelihood** contributions from
two models fitted to the same independent sampling units. The fail-closed
default relation is `unknown`.

## Rust-computed statistic

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

The uncorrected normal selection statistic is

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

Positive raw values point toward model A and negative raw values toward model
B. They are not, by themselves, permission to report a preference. The Python
module performs no likelihood, variance, correction, standardization, or
probability calculation; those quantities come from `fast_mlsirm._core`
through `fast_mlsirm.fitstats.vuong_nonnested`.

## Two-stage inference contract

Vuong model selection requires two distinct questions.

1. **Are the fitted models distinguishable in the population?** The formal
   null of observational equivalence uses a weighted chi-square distribution
   for `N * omega_hat^2`. Its weights depend on casewise score vectors and
   information matrices.
2. **If distinguishable, which model is closer in expected log likelihood?**
   The normal `Z` statistic above addresses this second question.

The repository currently implements the second statistic but does not yet
expose the score-vector and information-matrix inputs required for the formal
first stage across every model family. Therefore this release never converts a
raw `Z` value into `MODEL_A_PREFERRED` or `MODEL_B_PREFERRED`.

## Fail-closed status sequence

1. Model labels are trimmed, limited to 128 printable characters, and must be
   distinct.
2. Each casewise iterable is consumed only up to the documented maximum of
   1,000,000 values; oversized and non-terminating iterables fail before an
   unbounded allocation.
3. Parameter counts must be non-negative integers; booleans and fractional
   values are rejected.
4. The Rust core computes `mean_diff`, `omega`, `z`, and the two-sided normal
   p value.
5. Exact zero variance is translated into the typed
   `VuongVarianceDegenerateError` boundary signal and returned as
   `variance_degenerate`.
6. `omega_tol` is a numerical stability floor only. It is **not** Vuong's
   formal distinguishability test.
7. `strictly_non_nested` and `overlapping` relations return
   `requires_distinguishability_test`.
8. `nested` and `boundary_nested` relations return
   `requires_likelihood_ratio` because an ordinary, mixture, or parametric
   bootstrap reference distribution is required.
9. `unknown` returns `unknown_relation`; the library does not infer nestedness
   from parameter counts.

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
)

print(result.status)
# ComparisonStatus.REQUIRES_DISTINGUISHABILITY_TEST

print(result.raw_z, result.raw_p_two_sided)
# Auditable Rust selection statistic; not yet a model preference.

print(result.preferred_model)
# None
```

The result preserves these audit fields:

- `raw_mean_loglik_difference`
- `omega`
- `variance_positive`
- `raw_z`
- `raw_p_two_sided`
- model labels and parameter counts
- declared relation and explicit status
- warning explaining the required next procedure

The interpreted `z` and `p_two_sided` fields remain `NaN`, and
`preferred_model` remains `None`, until a future typed formal-
distinguishability result is integrated.

## Scope boundaries

### Independent observations

The normal calibration assumes the supplied casewise contributions are
independent sampling units. Repeated claims, judges, prompt variants, or turns
from the same query must not be passed as independent cases without a justified
sampling model. Cluster aggregation and bootstrap resampling are intentionally
absent rather than being implemented ad hoc in Python. A future cluster-robust
extension must be a Rust kernel with a documented asymptotic contract and
recovery study.

### Nested and boundary cases

Ordinary nested models require the relevant likelihood-ratio reference
distribution. Parameters on a boundary, such as a variance fixed at zero, can
require a mixture distribution or parametric bootstrap. The API reports this
state instead of applying the non-nested normal reference distribution.

### Required follow-up

A complete decision API requires all supported fit families to expose, on one
common case unit:

- casewise score vectors;
- observed and expected information matrices;
- parameter-order metadata;
- nestedness and boundary metadata; and
- cluster identifiers where independence is not justified.

Those inputs must feed a Rust weighted-chi-square distinguishability kernel and
a recovery study before a public model-preference status is enabled.

## References

- Vuong, Q. H. (1989). Likelihood ratio tests for model selection and
  non-nested hypotheses. *Econometrica, 57*(2), 307–333.
  https://doi.org/10.2307/1912557
- Schneider, L., Chalmers, R. P., Debelak, R., & Merkle, E. C. (2020). Model
  selection of nested and non-nested item response models using Vuong tests.
  *Multivariate Behavioral Research, 55*(5), 664–684.
  https://doi.org/10.1080/00273171.2019.1664280
- Merkle, E. C., You, D., & Preacher, K. J. (2016). Testing nonnested
  structural equation models. *Psychological Methods, 21*(2), 151–163.
  https://doi.org/10.1037/met0000038
