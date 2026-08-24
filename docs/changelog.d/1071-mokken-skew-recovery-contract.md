# Correct skewed-population Mokken study contract

## Fixed

- Keep the normal-trait Monte Carlo condition as the calibrated H/recovery
  contract.
- Standardize the positive-skew half-normal latent condition to the same
  location and scale as the normal condition before applying the shared 1.5
  theta scale, so the study changes distribution shape without confounding
  skewness with the previous approximately 28% narrower latent spread.
- Require both moment-matched latent conditions to retain the calibrated
  Loevinger H band, while keeping AISP full-recovery acceptance calibrated on
  the normal condition rather than treating the user-selected `c = 0.3`
  cutoff as distribution-invariant.
- Preserve the exact ignored-study execution and report failures normally.
- Declare that the workflow consumes no secrets and require reviewed
  `${{ secrets.NAME }}` environment injection for any future credentialed
  study.
