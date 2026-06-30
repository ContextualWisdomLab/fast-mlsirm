# Fit Diagnostics Plan

## Paper Basis

Kang and Jeon (2025) and Jeon et al. (2021) motivate MLS2PLM/LSIRM fit
checking through latent-space response probabilities and posterior predictive
comparisons. Those paper-level posterior predictive checks require posterior
samples or replicated response matrices.

The current `fast-mlsirm` backend is a regularized JML/MAP-style optimizer,
closer to Molenaar and Jeon (2026). Until the package has a sampler or
posterior draw contract, fit diagnostics should be labeled as point-estimate
residual diagnostics, not posterior predictive model checks.

## Implemented Scope

`fit_diagnostics` computes deterministic diagnostics from fitted parameters:

- item-level observed count, observed score, expected score, raw residual,
  standardized residual, infit mean-square, and outfit mean-square
- person-level versions of the same statistics
- model-level log-likelihood, deviance, AIC, BIC, observed mean, expected mean,
  mean absolute residual, and Pearson chi-square

AIC and BIC use the active stored parameter count for the selected model. They
are descriptive for this growing-parameter JML setting and should not be
described as Bayesian posterior predictive fit.

## Deferred Scope

Posterior predictive itemfit, personfit, and model-fit checks are deferred until
the package has posterior samples or replicated response matrices. S-X2 style
grouped item-fit tests and p-values are also deferred because they require
additional grouping rules and distributional calibration.
