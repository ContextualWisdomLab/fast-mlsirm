# ADR-0008: True-parameter recovery is core scientific CI evidence

Status: **Accepted**  
Date: 2026-08-09

## Context

A numerical psychometric implementation can produce plausible-looking estimates and high correlations while being systematically biased, on the wrong scale, overconfident, or unstable near important parameter regions. In AI evaluation, human raw scores also contain rater and task effects; correlation with them is not a sufficient accuracy or validity claim.

The repository already uses simulation/recovery, Rust/NumPy parity and literature-design studies. This ADR makes the evidence hierarchy explicit.

## Decision

For estimators with known generating parameters, the default scientific acceptance evidence is:

1. identify and align the model scale/rotation/linking;
2. calculate parameter bias;
3. calculate MAE and/or RMSE;
4. evaluate SE bias and nominal interval coverage when uncertainty is exposed;
5. evaluate convergence/failure rate;
6. evaluate model-specific function recovery such as response probabilities, thresholds, information, distances or factor/loadings after appropriate alignment;
7. evaluate CPU/GPU/reference parity where multiple execution paths implement the same contract.

Correlation may be reported as supplementary order-preservation evidence but is not an accuracy gate.

### Scale/identification rule

Raw RMSE is invalid when parameters are unidentified up to scale, sign, permutation, rotation, reflection or translation. The recovery harness must apply the same identification or accepted alignment used for interpretation before error metrics.

Examples:

- latent-space positions -> Procrustes/aligned positions or distance matrices;
- multidimensional loadings -> sign/permutation/rotation alignment;
- linked IRT scales -> fixed anchors or a documented linking transform;
- rater severity -> identified centering/reference constraints.

### CI strategy

PR CI should contain bounded sentinel/recovery tests that catch scientific regressions without exhausting the merge queue. Expensive paper-design Monte Carlo studies remain scheduled/manual/release evidence with deterministic manifests proving coverage of the intended study inventory.

Studies whose scientific sample size cannot finish inside the generic 1,800-second ignored-shard subprocess budget run in a dedicated scheduled job. The multidimensional Graded Response Model 500-replication recovery (`mc_grm_recovery_500`) is one such lane: it is excluded from the 12-way shard inventory by a target-qualified name, executed exactly once under a 120-minute job ceiling, and published as a 90-day log artifact so bias, RMSE, convergence, and theta-correlation lines survive after the Actions log expires. Do not copy that job into pull-request CI.

Monte Carlo acceptance rules are chosen prospectively from theory/sampling precision, not retrofitted to one observed seed outcome.

## Consequences

Benefits:

- catches errors hidden by correlation;
- makes numerical changes reviewable against scientific consequences;
- supports Rust/GPU evolution without trusting implementation resemblance alone.

Costs:

- realistic simulations are computationally expensive;
- recovery thresholds require method-specific justification;
- some models need careful alignment before metrics are meaningful.

## Alternatives considered

- **Correlation-only validation.** Rejected because positive affine bias can preserve correlation perfectly.
- **Golden file of one fit.** Rejected as insufficient; deterministic regression is useful but cannot establish recovery across data-generating conditions.
- **Run every heavy study on every PR.** Rejected for queue/schedulability reasons; the studies are retained in scheduled/release evidence rather than deleted.

## References

Bland, J. M., & Altman, D. G. (1986). Statistical methods for assessing agreement between two methods of clinical measurement. *The Lancet, 327*(8476), 307–310.

Svetina, D., Valdivia, A., Underhill, S., Dai, S., & Wang, X. (2017). Parameter recovery in multidimensional item response theory models under complexity and nonnormality. *Applied Psychological Measurement, 41*(7), 530–544.
