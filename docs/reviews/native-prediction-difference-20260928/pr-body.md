## Proposal
Difference of two caller-declared linear mean predictions using one native covariance contrast. Reuse linear_contrast; native owns row subtraction. Finite row length, existing covariance/positive variance/overflow contracts apply. Not future-observation prediction intervals.

## Sources and validation
Docstrings cite the actually opened official statsmodels RegressionResults.t_test manual, r_matrix/cov_p/use_t and Rb=q. Native30 tests passed, exit0, using independent covariance algebra and reversed/invalid/zero-variance/overflow checks. Study consumer4 tests passed; H2/H3 low-DT high-minus-low-E level contrast matches independent estimate/SE oracle. Build/JUnit/source/core hashes retained here.

## Remaining acceptance
Current-head required CI, nonauthor review, normal integration, immutable release/install hashes and actual study input/fit provenance. No study output gate removed.
