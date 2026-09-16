# Single-group polytomous bifactor GRM with Gibbons-Hedeker reduction (stage 1 of #1912)

## Added

- Add a single-group full-information polytomous bifactor graded response
  fitter (Gibbons et al., 2007; Gibbons & Hedeker, 1992; Samejima, 1969):
  unconstrained general/specific slopes, strictly decreasing boundary
  intercepts, caller-supplied item-to-specific map with general-only items,
  Bock-Aitkin EM with Gibbons-Hedeker dimension reduction, deterministic
  multi-start selection, general-factor EAP scores with posterior SDs, and a
  `fit_bifactor_grm` Python binding. Unobserved categories fail loudly;
  `max_iter` exhaustion reports `converged=False` instead of substituting
  values. Validated against `mirt::bfactor(itemtype="graded")` on a committed
  fixture (loglik gap 0.015, slope gap <= 0.046, intercept gap <= 0.028).
