# Non-nested model comparison

`fast_mlsirm.model_comparison.compare_nonnested_models` compares two fitted IRT-family models from paired casewise marginal log-likelihood contributions.

## Decision sequence

1. Declare the mathematical relation: `nested`, `strictly_non_nested`, `overlapping`, `boundary_nested`, or `unknown`.
2. Aggregate dependent cells to an independent unit with `cluster_id`.
3. Check the numerical distinguishability prerequisite, `sd(loglik_a - loglik_b) > omega_tol`.
4. Run the Rust-backed Vuong non-nested z statistic with no correction, AIC correction, or BIC/Schwarz correction.
5. Report a deterministic cluster-bootstrap percentile interval for the corrected mean log-likelihood difference.
6. Use held-out query, system, domain, or judge-family likelihoods as a separate predictive-validity analysis.

The API deliberately does **not** label the variance prerequisite as Vuong's formal distinguishability hypothesis test. The formal weighted-chi-square test requires score and information matrices and remains separate future Rust work.

## Example

```python
from fast_mlsirm.model_comparison import compare_nonnested_models

result = compare_nonnested_models(
    loglik_a=bifactor_casewise_loglik,
    loglik_b=correlated_mirt_casewise_loglik,
    k_a=bifactor_parameter_count,
    k_b=mirt_parameter_count,
    relation="strictly_non_nested",
    cluster_id=query_id,
    correction="bic",
    bootstrap=2000,
    seed=20260803,
)

print(result.z, result.p_two_sided, result.preferred_model)
print(result.bootstrap_ci)
```

## Clustering guidance

- Use `query_id` when several claims, probes, judges, or prompt variants share one RAG question.
- Use `system_id` when the intended sampling unit is a RAG configuration and repeated runs are subordinate observations.
- Use `judge_family_id` for leave-family-out or family-block sensitivity analysis.
- Do not treat claim-level cells from the same answer as independent cases.

## Model-relation cautions

- `correlated MIRT` versus `bifactor` is generally non-nested or overlapping and is a suitable Vuong comparison after distinguishability is established.
- `bifactor` versus `second-order` is non-nested.
- A latent-space extension with `gamma = 0`, a testlet variance of zero, or a judge variance of zero is boundary-nested. Prefer a parametric-bootstrap likelihood-ratio test as the primary inferential procedure.
- If models are not distinguishable or the Vuong result is not significant, retain the simpler, more interpretable model unless predictive evidence strongly favors the alternative.

## References

Schneider, L., Chalmers, R. P., Debelak, R., & Merkle, E. C. (2020). Model selection of nested and non-nested item response models using Vuong tests. *Multivariate Behavioral Research, 55*(5), 664-684. https://doi.org/10.1080/00273171.2019.1664280

Vuong, Q. H. (1989). Likelihood ratio tests for model selection and non-nested hypotheses. *Econometrica, 57*(2), 307-333. https://doi.org/10.2307/1912557
