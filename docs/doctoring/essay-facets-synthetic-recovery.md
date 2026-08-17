# Essay facets synthetic recovery evidence

## Status and scope

This note governs the deterministic recovery study in
`tests/test_scoring_essay_facets_synthetic_recovery.py`. The study is
**validation evidence**, not a new estimator. Production likelihood,
quadrature, optimization, and parameter estimation remain owned by the Rust
many-facet kernel and are reached through the governed scoring-facets Python
orchestration boundary.

The test intentionally treats rater scores as fallible observations. It does
not treat either a human or automated scorer as ground truth by identity and
does not claim that agreement or correlation alone establishes validity.

## Model traced by the test

For respondent ability \(\theta_p\), task difficulty \(\delta_i\), rater
severity \(c_r\), and shared ordered-category thresholds \(\tau_m\), the
simulation uses the repository's rating-scale many-facet form:

\[
\log \frac{P(Y_{pir}=k)}{P(Y_{pir}=k-1)}
= \theta_p - \delta_i - c_r - \tau_k.
\]

Equivalently, with \(T_k=\sum_{m=1}^{k}\tau_m\), the category logit used by
the test-only generator is

\[
\psi_k = k\theta_p-k(\delta_i+c_r)-T_k,
\qquad
P(Y_{pir}=k)=\operatorname{softmax}(\psi)_k.
\]

This matches the parameter roles used by the repository Rust `fit_facets`
implementation and the essay-writing many-facet formulation summarized by Uto
and Aramaki (2024): examinee ability, writing-task difficulty, rater severity,
and ordered step parameters are distinct model components.

## Identification and alignment

The recovery fixture uses centered generating task difficulties, centered
rater severities, and thresholds summing to zero. Task, rater, and respondent
estimates are compared after aligning them to the governed task-revision,
rater-engine, and respondent axes returned by the assembled calibration
design. The governed `allowed_scores` contract requires the category values to
be sorted and unique. `fit.thresholds` exposes the corresponding \(K-1\)
ordered step values without a separate threshold-axis label, so the generated
and fitted threshold vectors are compared positionally in that shared category
step order. The test therefore does not infer task, rater, or respondent
identity from insertion order, display labels, or a scorer's human/AI class,
and it does not imply that an unlabelled threshold vector can be realigned by
an independent axis that the fit result does not provide.

The bounded study reports test-layer bias, MAE, and RMSE for:

- rater severity;
- task difficulty; and
- respondent EAP standing.

For shared category thresholds it separately verifies the identified sum-zero
constraint and reports MAE and RMSE. Mean signed threshold bias is not used as
independent recovery evidence because both the generating and fitted threshold
vectors are constrained to sum zero; under that identification, the mean
signed difference is algebraically near zero even when individual thresholds
are poorly recovered.

Rater-severity and task-difficulty ordering are checked separately. Correlation
is deliberately not used as the primary acceptance criterion because a high
correlation can coexist with substantial scale or location error. Threshold
recovery is checked on the identified sum-zero scale rather than inferred from
category-use correlation or support alone.

## Current acceptance boundary

The deterministic fixture uses a fully crossed design with four task revisions,
two raters, three ordered categories, and 500 respondents. The exact numerical
bounds are regression gates for this seeded fixture, not universal psychometric
quality cutoffs. If a current-head run exceeds a bound, the response is to
inspect the estimator, identification, simulation, and sampling behavior; the
bound must not be loosened merely to obtain a green check.

This slice does **not** establish:

- rater consistency/discrimination recovery;
- rater-specific threshold or range-restriction recovery;
- uncertainty or interval coverage for rater/task/threshold parameters when the
  current estimator does not expose the corresponding standard errors;
- subgroup fairness, DIF, invariance, or multilingual comparability;
- construct validity or score interpretation;
- human/automated scorer interchangeability; or
- authorization for consequential or high-stakes automated scoring.

Those claims require separately identified models and their own recovery,
coverage, fairness, validity, and operational evidence.

## Research traceability

Uto and Aramaki (2024) use many-facet models in an essay-writing setting and
explicitly distinguish examinee ability, task difficulty, rater severity, and
step parameters. Their work also discusses richer rater effects such as
consistency and range restriction in generalized models. The present test uses
only the simpler severity baseline already implemented by `fast-mlsirm`; it
does not collapse those richer effects into the severity parameter.

### APA 7th references

Uto, M., & Aramaki, K. (2024). Linking essay-writing tests using many-facet
models and neural automated essay scoring. *Behavior Research Methods, 56*,
8450–8479. https://doi.org/10.3758/s13428-024-02485-2

Uto, M., & Ueno, M. (2020). A generalized many-facet Rasch model and its
Bayesian estimation using Hamiltonian Monte Carlo. *Behaviormetrika, 47*,
469–496. https://doi.org/10.1007/s41237-020-00115-7
