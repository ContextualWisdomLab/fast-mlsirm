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

## User-Facing Contract

The diagnostic flow is:

1. Choose the response process: `cumulative` or `ideal_point`.
2. Choose the item type: `dichotomous` or `polytomous`.
3. Fit or supply a model that returns category probabilities.
4. Run diagnostics over those probabilities.

The CLI mirrors the API:

```bash
fast-mlsirm diagnose-response-process \
  --responses responses.npy \
  --probabilities probabilities.npy \
  --item-type polytomous \
  --response-process cumulative \
  --out runs/process_fit
```

## What This PR Supports

- Shared dichotomous/polytomous item, person, category, and model fit summaries
  from category probabilities.
- Existing MLS2PLM binary diagnostics remain point-estimate diagnostics.
- Dimensionality diagnostics remain K-fold held-out likelihood diagnostics for
  the current JML/MAP backend.

## Deferred

- Estimation for GRM, GPCM, GGUM, or multidimensional GGUM.
- Posterior predictive checking until posterior samples or replicated response
  matrices exist.
- Calibrated p-values for limited-information item-pair or item-triple fit.
