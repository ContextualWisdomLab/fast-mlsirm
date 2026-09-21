# Bifactor GRM opt-in lognormal |a| slope prior

## Added

- `fit_bifactor_grm` and `fit_bifactor_grm_multigroup` accept paired
  `slope_prior_mu` / `slope_prior_sd` for MAP estimation under a lognormal
  prior on `|a|` (folded lognormal on signed slopes; no defaults). In the
  multigroup fit the prior applies to free items per group AND to common
  (anchored) items once per shared parameter, so it is active under the
  default `anchor=None`. FIPC exposes no prior knob. Results record the
  fitted prior (`slope_prior_mu` / `slope_prior_sd`). Under a prior the EM
  monotonicity guard, `tol` convergence, `final_loglik_change` and
  multi-start ranking use the log posterior; `loglik_trace` stays the
  observed-data log-likelihood.
- `bifactor_oakes_se` accepts the same prior. With it, `information` is the
  negative log-posterior curvature (Oakes observed information plus the
  diagonal prior curvature on slopes) and `vcov`/`se` are a posterior-curvature
  (Laplace) approximation to the posterior covariance, not a frequentist
  sampling covariance (Mislevy, 1986, https://doi.org/10.1007/BF02293979).
  Omitting the prior remains the MML observed-information SE.
