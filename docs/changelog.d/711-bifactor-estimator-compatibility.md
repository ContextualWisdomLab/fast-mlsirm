# Bifactor estimator compatibility preflight

## Fixed

- Reject unsupported `BIFAC2PLM` plus `jmle` configurations at `FitConfig`
  validation and preserve `mmle` as the supported bifactor estimator.
- Add regression coverage for the complete current public model–estimator
  compatibility matrix and APA 7th doctoring.
