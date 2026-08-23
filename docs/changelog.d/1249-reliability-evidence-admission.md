# Reliability evidence admission

## Fixed

- Reject callback-bearing, complex, or non-real-numeric caller evidence before Rust discovery in Guttman lambda, ten Berge mu, Cronbach alpha, and person-separation reliability entry points, while preserving ordinary NumPy arrays and trusted built-in sequence inputs.
- Reject over-nested or cyclic built-in sequence evidence at the public API's known 1-D/2-D rank boundary before NumPy materialization or native discovery, while preserving shared acyclic rows and trusted real-scalar sequence compatibility.
- Use one callback-free masked-array diagnostic across ICC, Guttman lambda, ten Berge mu, Cronbach alpha, person separation, and pairwise-rater reliability so masked evidence consistently tells callers to encode missingness with NaN before any native dispatch.
- Preserve historical built-in sequence compatibility when rows are exact real-numeric NumPy arrays, while retaining callback-free rejection of ndarray subclasses and non-real row storage before materialization.
- Preserve historical rater-sequence Boolean semantics without reopening caller protocols: pure Boolean built-in sequences keep the Boolean-specific diagnostic, while mixed Boolean+numeric built-in sequences retain NumPy's numeric promotion.
- Make reliability-adapter installation recover every primary sibling after an interrupted partial bind instead of treating a hardened ICC wrapper alone as proof that the whole public reliability surface was installed.
- Bound primary and rater reliability evidence to 20,000,000 logical cells before NumPy materialization or contiguous `float64` allocation, including exact broadcast views and exact NumPy leaves nested inside trusted built-in sequences.
