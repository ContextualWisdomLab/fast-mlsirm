# Governed criterion-level facets calibration handoff

## Added

- Added factory-sealed, content-addressed scoring-facets rating, design, and calibration-bundle contracts that project exact governed scoring requests, results, engines, respondents, tasks, criteria, terminal states, score scales, and engine fingerprints into criterion-specific many-facet designs.
- Added fail-closed provenance, duplicate-cell, observed-support, category-identification, dense-allocation, and task-rater connectedness gates. Abstained, failed, excluded, and absent cells remain missing rather than being coerced to low scores, while ordered rubric categories are mapped to zero-based estimator categories only at the Rust boundary.
- Added `fit_scoring_facets_bundle`, which delegates likelihood, EM, quadrature, and parameter updates to the existing Rust-backed `fit_facets` implementation and calibrates analytic criteria separately instead of averaging them. The handoff makes no convergence, fit, reliability, fairness, scoreability, construct-validity, rater-interchangeability, or high-stakes automation claim.
