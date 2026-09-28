# Two-Tier FIPC Class A Fix

The canonical `fc937dd6` trace showed the direct-GH observed-data log-likelihood
falling at EM iteration 2. Production FIPC intentionally remaps the fixed
standard-normal Gauss-Hermite nodes through the focal mean and Cholesky factor,
keeps the standard weights, and scales specific nodes by the focal specific SD;
the failure came from applying item M-step updates and then moving that mapped
support with the focal moment update without checking the combined remapped
objective.

The direct-GH path now evaluates the candidate focal-prior update with the
updated item parameters on the same remapped nodes and standard weights. If the
candidate regresses, it backtracks the mean, covariance, and specific scales;
if no positive step is acceptable, it restores the item parameters and prior
state together. This keeps E-statistics, the item M-step, prior moments, and
the scoring/EAP pass on one coordinate system and leaves the strict LL guard
unchanged.

Non-finite candidate LLs are rejected explicitly before line search; NaN is not
allowed through comparison semantics. A complete rollback is marked as a
non-convergent iteration, so the flat LL produced by restoring the prior state
cannot be mistaken for EM convergence on the following pass.

Anchored item rows remain bit-for-bit fixed. Non-unit focal priors are represented
only by mapped primary nodes and scaled specific nodes, never by density-ratio
weights. Final scoring repeats the same direct-GH mapping and standard weights,
so reported EAPs use the measure used during fitting.

When a full prior update fails the direct-GH likelihood check, backtracking now
interpolates the free item parameters and focal prior together from the previous
accepted state. Shrinking only the prior while retaining the full item M-step
candidate tested an inconsistent pair and caused every trial to roll back; the
joint step preserves the E-step/M-step coordinate relationship while allowing a
small monotone update toward the non-unit focal fixture.

To prevent an otherwise unbounded direct-GH retry loop, three consecutive full
rollbacks terminate with `termination_reason="prior_update_stalled"` and
`converged=false`. The result reports accepted prior steps, full rollback count,
and the final rollback streak; these counters are diagnostic only and do not
change the strict likelihood guard or tolerance.
