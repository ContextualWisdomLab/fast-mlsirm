# Achieved finite-population proportion research basis

## Product boundary

ADR 0023 is normative. This primitive terminates a completed one-stratum
SRSWOR design from ADR 0022. It does not select the sample, classify source
content, replace failed units, or authorize a downstream population claim.

## Adopted evidence

Cochran's SRSWOR design variance retains the finite-population sampling
fraction and estimates the binary population variance from the completed
sample. Wang (2015) studies exact confidence intervals for the hypergeometric
parameter `M`, the count possessing an attribute in a known finite population.
The adopted equal-tailed `C_O` interval is the Konijn interval described in
that paper. Its coverage is conservative on the discrete parameter grid and
is verified here by exhaustive enumeration.

The implementation does not adopt Wang's Algorithm II and therefore makes no
claim of set-inclusion admissibility or optimality. Exact means minimum
coverage at least the declared level, not equality at every `M`.

## References

Cochran, W. G. (1977). *Sampling techniques* (3rd ed.). John Wiley & Sons.

Wang, W. (2015). Exact optimal confidence intervals for hypergeometric
parameters. *Journal of the American Statistical Association, 110*(512),
1491–1499. https://doi.org/10.1080/01621459.2014.966191
