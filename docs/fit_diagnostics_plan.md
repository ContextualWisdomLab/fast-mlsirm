# Fit Diagnostics Plan

## Paper Basis

Kang and Jeon (2025) and Jeon et al. (2021) motivate MLS2PLM/LSIRM fit
checking through latent-space response probabilities and posterior predictive
comparisons. Those paper-level posterior predictive checks require posterior
samples or replicated response matrices.

The current `fast-mlsirm` backend is a regularized JML/MAP-style optimizer,
closer to Molenaar and Jeon (2026). Their JML framing supports selecting the
latent-space dimension with cross-validation. Until the package has a sampler
or posterior draw contract, fit diagnostics should be labeled as point-estimate
residual diagnostics, not posterior predictive model checks.

## Implemented Scope

`fit_diagnostics` computes deterministic diagnostics from fitted parameters:

- item-level observed count, observed score, expected score, raw residual,
  standardized residual, infit mean-square, and outfit mean-square
- person-level versions of the same statistics
- factor-level versions of the same statistics for multidimensional models
- model-level log-likelihood, deviance, AIC, BIC, observed mean, expected mean,
  mean absolute residual, and Pearson chi-square

`dimensionality_diagnostics` fits candidate latent-space dimensions with
K-fold held-out entries and reports validation log-likelihood, deviance, mean
absolute residual, and RMSE. The selected dimension is the candidate with the
largest held-out log-likelihood.

`response_process_fit_diagnostics` accepts an observed response matrix and
model-provided category probabilities. This separates response process from
diagnostic aggregation:

- dichotomous cumulative or ideal-point models can pass `N x J` success
  probabilities or `N x J x 2` category probabilities
- polytomous cumulative models such as GRM/GPCM and ideal-point/unfolding
  models such as GGUM can pass `N x J x K` category probabilities
- diagnostics aggregate item, person, category, and model fit without claiming
  that `fast-mlsirm` currently estimates every response-process family

AIC and BIC use the active stored parameter count for the selected model. They
are descriptive for this growing-parameter JML setting and should not be
described as Bayesian posterior predictive fit.

## Deferred Scope

Posterior predictive itemfit, personfit, and model-fit checks are deferred until
the package has posterior samples or replicated response matrices. S-X2 style
grouped item-fit tests and p-values are also deferred because they require
additional grouping rules and distributional calibration.
