# Angoff delta-plot DIF screen

`fast_mlsirm.delta_plot` / `mlsirm_core::dif::delta_plot` implements Angoff's
transformed-item-difficulty (TID) / delta-plot procedure for dichotomous
observed-score differential item functioning (DIF). It is a psychometric
item-screen, not a security control. CWE, OWASP, and NIST AI/risk publications
are not the methodological basis for this estimator.

The procedure is distinct from the Mantel–Haenszel, logistic-regression, and
SIBTEST screens already documented in
[`rubric_dif_pilot_handoff.md`](rubric_dif_pilot_handoff.md). Those methods
remain available; this page records only the delta-plot path.

Architecture decision: [`adr/0016-angoff-delta-plot-dif.md`](adr/0016-angoff-delta-plot-dif.md).

## What is implemented

For a persons-by-items matrix containing `0`, `1`, or `NaN` values and a
two-group coding (`0` = reference, `1` = focal):

1. Compute the per-item proportion correct in each group. Missing cells
   (`NaN`) are dropped per item per group; values other than `0`, `1`, or
   `NaN` are rejected.
2. Adjust extreme proportions (`constraint` clamp or `add` pseudo-counts).
3. Transform adjusted proportions to ETS-style delta difficulties

   ```text
   Delta = 4 * qnorm(1 - p) + 13
   ```

4. Fit the major axis of the reference/focal delta cloud and compute each
   item's perpendicular distance from that axis.
5. Flag items whose absolute distance exceeds either the Magis and Facon
   (2012) normal-approximation threshold at `alpha` (`threshold="norm"`) or a
   fixed threshold (`threshold="fixed"`; classical default `1.5`).
6. Optionally purify the axis by iteratively dropping flagged items
   (`IPP1` / `IPP2` / `IPP3`). A fixed threshold forces IPP1 semantics.

Python validates arrays and marshals the result. All numeric work is in the
Rust kernel. There is no NumPy formula fallback for this screen.

## Interpretation boundary

A flagged item is a candidate for review. The screen does not show that:

- a flagged item is unfair or causally biased;
- an unflagged item is invariant;
- the matching/total-score comparison is uncontaminated;
- impact and DIF are separated as in a latent-variable DIF model;
- the sample has adequate focal/reference support; or
- generated or operational items are suitable for high-stakes use.

Fairness and score-use decisions remain governed by the *Standards for
Educational and Psychological Testing* (AERA, APA, & NCME, 2014) and by
triangulation with theory, calibrated DIF where justified, and human review.

## Claims not made

- This is not Mantel–Haenszel, logistic DIF, or SIBTEST.
- This is not a multiple-group IRT or bifactor DIF model.
- Package documentation or a computational port of `deltaPlotR` is a
  numerical comparison source, not a substitute for the primary papers.
- Changelog notes that a paper was unread at implementation time are
  historical source-governance comments, not a reason to omit the method's
  bibliographic basis.

## Public Python API

```python
import numpy as np
from fast_mlsirm import delta_plot

responses = np.array(
    [
        [1, 1, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ],
    dtype=np.float64,
)
group = np.array([0, 0, 1, 1], dtype=np.uint8)

result = delta_plot(responses, group, threshold="norm", alpha=0.05)
print(result.dif_items)
print(result.deltas)
```

## Primary sources

Angoff, W. H. (1972, September). *A technique for the investigation of
cultural differences* [Paper presentation]. Annual meeting of the American
Psychological Association, Honolulu, HI, United States.

Angoff, W. H., & Ford, S. F. (1973). Item-race interaction on a test of
scholastic aptitude. *Journal of Educational Measurement, 10*(2), 95–105.
https://doi.org/10.1111/j.1745-3984.1973.tb00787.x

Magis, D., & Facon, B. (2012). Angoff's Delta method revisited: Improving DIF
detection under small samples. *British Journal of Mathematical and
Statistical Psychology, 65*(2), 302–321.
https://doi.org/10.1111/j.2044-8317.2011.02025.x

Magis, D., & Facon, B. (2014). deltaPlotR: An R package for differential item
functioning analysis with Angoff's Delta Plot. *Journal of Statistical
Software, 59*(1), 1–19. https://doi.org/10.18637/jss.v059.c01

American Educational Research Association, American Psychological
Association, & National Council on Measurement in Education. (2014).
*Standards for educational and psychological testing*. American Educational
Research Association.
