# Response-time EM iteration ceiling

## Summary

The lognormal response-time measurement model runs a marginal-ML EM for item
time intensity/discrimination and population speed SD. Unbounded `max_iter`
exposes a denial-of-service path for direct `_core.fit_rt_lognormal` callers that
bypass Python validation. The Rust core now rejects values outside
`1..=100_000`, matching the package-wide Python `MAX_MAX_ITER` contract used by
other estimators.

## References

van der Linden, W. J. (2007). A hierarchical framework for modeling speed and
accuracy on test items. *Psychometrika, 72*(3), 287–308.
https://doi.org/10.1007/s11336-006-1478-z
