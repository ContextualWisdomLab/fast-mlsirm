# Finite-population proportion sampling design research basis

## Product boundary

This is a reusable measurement-design primitive owned by `fast-mlsirm`, not a
TEPP temporal-relational estimator and not a downstream product threshold.
ADR 0022 is normative; this note records its external evidence.

## Adopted evidence

NIST connects a caller-selected confidence level, absolute sampling error, and
expected population proportion through the normal approximation. For sampling
without replacement, NIST applies the finite-population correction. NIST also
notes that `p=0.5` maximizes the required sample size when no prior proportion
is known; the library does not turn that observation into a default because
the caller must retain the source of every assumed proportion.

ABS requires strata to be exhaustive, non-overlapping, and independently
sampled. It distinguishes proportional allocation, used when stratum
variability is unavailable or similar, from optimum allocation, which assigns
more observations to more variable strata and may incorporate costs. The v1
contract adopts equal-cost Neyman allocation only; unequal costs remain
unavailable rather than receiving an arbitrary cost weight.

## Explicit exclusions

- No design-effect or nonresponse multiplier is inferred.
- No sample fraction threshold decides whether FPC applies; the without-
  replacement contract always reports it.
- No exact-binomial, rare-event, cluster, multistage, or multi-objective claim
  is made by the normal-approximation artifact.
- No integer minimum is silently assigned to a stratum whose formula receives
  zero; the whole design fails and asks the caller to revise its evidence.

## References

Australian Bureau of Statistics. (2023, March 1). *Sample design*. https://www.abs.gov.au/websitedbs/D3310114.nsf/home/Basic%20Survey%20Design%20-%20Sample%20Design

National Institute of Standards and Technology. (n.d.). *Sample sizes
required*. In *NIST/SEMATECH e-Handbook of statistical methods*. Retrieved
August 26, 2026, from
https://www.itl.nist.gov/div898/handbook/prc/section2/old.prc272.htm

National Institute of Standards and Technology. (n.d.). *Confidence limits*.
In *NIST/SEMATECH e-Handbook of statistical methods*. Retrieved August 26,
2026, from
https://www.itl.nist.gov/div898/handbook/prc/section2/old.prc271.htm
