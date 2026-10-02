# Multigroup bifactor Oakes information under a slope prior (#2113 phase 2)

## Added

- `bifactor_multigroup_oakes_se` now honors a MAP fit. When the fit records
  `slope_prior_mu` / `slope_prior_sd`, the joint information adds the diagonal
  lognormal `|a|` prior curvature on each estimated slope: common items once,
  free items once per group, and none on thresholds or group distributions.
  `information` is then the negative log-posterior curvature, and `vcov`/`se`
  are a Laplace approximation to the posterior covariance (Mislevy, 1986,
  https://doi.org/10.1007/BF02293979), not a sampling covariance. Fits without
  a prior return the unchanged ML information.
