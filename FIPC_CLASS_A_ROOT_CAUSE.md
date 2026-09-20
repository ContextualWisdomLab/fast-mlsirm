# Two-Tier FIPC Class A Fix

The canonical `fc937dd6` run showed a true observed-data log-likelihood decrease
when both iterations were evaluated on one frozen standard-normal quadrature
measure (`-1404.6909, -1405.8755, -2221.4383`). The production path instead
mapped Gauss-Hermite node locations to the current focal mean/covariance and
specific standard deviation, then consumed those transformed-node statistics in
the item and focal-prior M-steps; comparing that moving-support objective with
the frozen diagnostic exposed the inconsistent finite-quadrature E/M measure.

The fix keeps the standard-normal node support fixed and applies the current
focal primary and specific priors as density-ratio log weights. The E-step,
item M-step, focal moment updates, monotonicity guard, and final EAP pass now
use the same support and measure, while the guard remains strict and no
likelihood tolerance is widened.
