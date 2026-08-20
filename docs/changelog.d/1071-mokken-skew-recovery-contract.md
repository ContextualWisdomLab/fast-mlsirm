# Correct skewed-population Mokken study contract

- Keep the normal-trait Monte Carlo condition as the calibrated H/recovery
  contract.
- Treat the skew-trait condition as a finite, distribution-sensitivity check
  instead of incorrectly requiring AISP's `c = 0.3` cutoff to recover every
  item for every observed trait distribution.
- Preserve the exact ignored-study execution and report failures normally.
- Declare that the workflow consumes no secrets and require reviewed
  `${{ secrets.NAME }}` environment injection for any future credentialed
  study.
