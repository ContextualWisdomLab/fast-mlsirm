# Preserve strict GRM threshold order during initialization

## Fixed

- Preserve strictly decreasing finite GRM category thresholds for sparse or collapsed observed-category patterns by using the existing positive-pseudocount cumulative frequencies directly instead of independently clipping adjacent cumulative probabilities onto the same boundary. Returned GRM fits therefore remain inside the shared scoring and prediction parameter domain without changing Samejima category-probability arithmetic or GPCM behavior.
