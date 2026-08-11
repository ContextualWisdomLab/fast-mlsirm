# Fail early for unimplemented estimator identities

## Fixed

- Restricted the public `FitConfig.estimator` vocabulary to the implemented `jmle` and `mmle` fitting paths, so unsupported `em` and `bayes` requests fail during configuration validation instead of entering a fitting path that later raises `NotImplementedError`.
