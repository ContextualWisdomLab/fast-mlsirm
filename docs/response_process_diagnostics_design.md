# Response Process Diagnostics Design

## Product Goal

Researchers need to compare dimensionality and model fit across binary and
ordered-category IRT models without confusing response-process families.
`fast-mlsirm` should make the model contract explicit before showing fit
numbers.

## Literature Grounding

- MLSIRM/MLS2PLM posterior predictive checks compare observed and replicated
  response summaries, which requires posterior draws.
- The current implementation is a regularized JML/MAP backend, so held-out
  likelihood and residual diagnostics are the supported runtime diagnostics.
- Tay et al. (2011) compare dichotomous and polytomous ideal-point and
  dominance models, so `item_type` and `response_process` should be first-class
  metadata.
- GGUM-style ideal-point models and GRM/GPCM-style cumulative models should
  feed category probabilities into a shared diagnostic aggregator.
- Fox and Glas (2001) motivate multilevel IRT as a model layer, while Bock and
  Zimowski (1997) motivate multiple-group IRT comparisons. Until those
  estimators exist here, `fast-mlsirm` should expose group and cluster fit
  summaries as diagnostics, not as fitted hierarchical parameters.

## User-Facing Contract

The diagnostic flow is:

1. Choose the response process: `cumulative` or `ideal_point`.
2. Choose the item type: `dichotomous` or `polytomous`.
3. Fit or supply a model that returns category probabilities.
4. Run diagnostics over those probabilities.
5. If the study has populations or nesting units, pass numeric `group_id` or
   `cluster_id` arrays with one value per person.

The CLI mirrors the API:

```bash
fast-mlsirm diagnose-response-process \
  --responses responses.npy \
  --probabilities probabilities.npy \
  --item-type polytomous \
  --response-process cumulative \
  --group-id group_id.npy \
  --cluster-id school_id.npy \
  --out runs/process_fit

fast-mlsirm diagnose-response-candidates \
  --responses responses.npy \
  --candidate dim1=prob_dim1.npy \
  --candidate dim2=prob_dim2.npy \
  --item-type dichotomous \
  --response-process ideal_point \
  --out runs/process_dimensions
```

## What This PR Supports

- Shared dichotomous/polytomous item, person, category, and model fit summaries
  from category probabilities.
- Group and cluster summaries for multigroup and multilevel-context diagnostics.
- Probability-candidate comparisons for dimensionality or response-process
  checks when the probabilities come from an external model.
- Existing MLS2PLM binary diagnostics remain point-estimate diagnostics.
- Dimensionality diagnostics remain K-fold held-out likelihood diagnostics for
  the current JML/MAP backend.

## Deferred

- Estimation for GRM, GPCM, GGUM, or multidimensional GGUM.
- Estimation of multiple-group or multilevel IRT parameters.
- Posterior predictive checking until posterior samples or replicated response
  matrices exist.
- Calibrated p-values for limited-information item-pair or item-triple fit.
