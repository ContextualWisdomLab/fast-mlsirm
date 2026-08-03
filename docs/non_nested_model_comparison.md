# Fail-closed Vuong selection-statistic summary

`fast_mlsirm.model_comparison.compare_nonnested_models` provides an auditable,
resource-bounded wrapper around the existing Rust-backed Vuong **selection
statistic**. It does not declare a winning model before the mathematically
required first-stage distinguishability test has been supplied.

The API accepts paired **casewise marginal log-likelihood** contributions from
two models fitted to the same independent sampling units. The fail-closed
default relation is `unknown`.

## Relation-first procedure routing

The caller-declared model relation is routed before a normal-selection statistic
is computed. The API does not infer or validate the mathematical relation;
omitted metadata defaults to `unknown`. Incorrect relation metadata can route a
comparison to the wrong statistical procedure.

- `nested` and `boundary_nested` return `requires_likelihood_ratio` and do not
  invoke the non-nested kernel.
- `unknown` returns `unknown_relation` and does not invoke the kernel.
- `strictly_non_nested` and `overlapping` may invoke the Rust selection-statistic
  kernel, but still return `requires_distinguishability_test` until the formal
  weighted-chi-square first stage is available.

This order prevents an exact-zero or otherwise rejected non-applicable statistic
from masking the procedure actually required by the declared model relation.
Raw Vuong fields are therefore `NaN` for nested, boundary-nested, and unknown
relations.

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

Positive `raw_mean_loglik_difference` values point toward model A and negative
values toward model B. With `bic_correction=False`, `raw_z` has the same
model-direction interpretation. With `bic_correction=True`, the BIC penalty can
make the sign of `raw_z` differ from the sign of
`raw_mean_loglik_difference`. `omega` and `raw_p_two_sided` are not directional.
None of these fields, by itself, authorizes a model preference. The Python
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
   1,000,000 values. Every value is converted to a finite float under a stable
   public validation boundary; booleans, opaque objects, non-finite values, and
   conversion overflows are rejected before FFI.
3. The two vectors must have equal length with at least two independent cases.
4. Parameter counts must be non-negative integers; booleans and fractional
   values are rejected.
5. Nested, boundary-nested, and unknown relations are routed to their required
   procedure without invoking the non-nested normal-selection kernel.
6. For strictly non-nested or overlapping relations, the Rust core computes
   `mean_diff`, `omega`, `z`, and the two-sided normal p value.
7. A stable conversion or compiled-kernel rejection is converted to the
   redacted typed `VuongKernelError` boundary and returned as `kernel_error`.
   The wrapper does not inspect or expose exception wording and does not guess
   whether the low-level cause was zero variance or another rejection.
8. A successful applicable kernel result with zero, non-finite, or numerically
   tiny `omega` is returned as `variance_degenerate`.
9. `omega_tol` is a numerical stability floor only. It is **not** Vuong's formal
   distinguishability test.
10. `strictly_non_nested` and `overlapping` relations return
    `requires_distinguishability_test` after a valid raw statistic is available.

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

The result preserves these audit fields when an applicable compiled kernel
returns successfully:

- `raw_mean_loglik_difference`
- `omega`
- `variance_positive`
- `raw_z`
- `raw_p_two_sided`
- model labels and parameter counts
- declared relation and explicit status
- warning explaining the required next procedure

For relation-inapplicable paths, raw numerical fields are `NaN` because the
kernel is deliberately not called. When an applicable compiled kernel rejects
an input, all raw numerical fields are also `NaN` and the public result is
`kernel_error`; no rejected values or exception text are copied into the
result. The interpreted `z` and `p_two_sided` fields remain `NaN`, and
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
a recovery study before a public model-preference status is enabled. A future
compiled structured error code may refine `kernel_error`, but this release does
not derive a subtype from message text.

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
