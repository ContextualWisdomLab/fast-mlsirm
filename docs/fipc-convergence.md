# Two-tier FIPC convergence contract

## Scope and source basis

This change corrects a stopping decision, not the likelihood, posterior moments,
quadrature rule, or item optimizer. The ordinary two-tier fit is unchanged.

Chalmers (2012, p. 6, Section 3.1) describes repeating EM until the change
between iterations falls below a specified tolerance. That passage does not
specify the exact parameter metric or a rule for damped updates. The contemporary
mirt implementation provides the parameter-change convention:
`hasConverged(p0, p1, TOL)` compares all absolute parameter differences with TOL,
with transformations for its guessing/upper-bound/proportion parameters.
The EM caller compares pre- and post-M-step parameters. These sources motivate
a parameter-change check; mirt is not a runtime dependency or validation oracle.

Pinned source references, read on September 30, 2026:

- [mirt parameter-change test](https://github.com/philchalmers/mirt/blob/4ef09d3e4b6903442ec61bb11f593b7418aa5290/R/utils.R#L2929-L2941).
- [mirt EM call site](https://github.com/philchalmers/mirt/blob/7dd085d61a6a378a6bc37442bb449c9190c68632/R/EMstep.group.R#L335-L345).

## Stopping rule

Convergence requires all three conditions for the preceding outer EM sweep:

1. Its joint item-and-prior candidate was accepted in full, without outer
   backtracking or partial mean/scale recovery.
2. The maximum full-candidate parameter displacement is at most `tol`:
   free item slopes/thresholds, focal means, log focal SDs, off-diagonal focal
   covariances, and log specific SDs when estimated. Nonfinite displacement
   cannot satisfy the check.
3. The absolute observed-data likelihood change is at most
   `tol * (1 + abs(previous_loglik))`.

The full candidate displacement is measured before outer damping or restoration,
not from the accepted shrunken step. A damped step cannot certify convergence,
even if both numerical changes happen to be small. It may continue improving;
when the iteration budget ends on such a step the result is nonconverged with
`step_limited`. Existing repeated-rollback termination remains
`prior_update_stalled`. `final_param_change` reports full-candidate displacement,
not the last accepted displacement. Internal Newton line search is unchanged;
full-step eligibility here refers to the outer joint EM acceptance decision.

## Reproduction and limits

The low-resolution seed-1000 bug reproduction uses 180 people, six four-category
items, two primary and two specific factors, and anchors at items 0, 1, 3, and 4.
Two item rows remain free. At seven quadrature nodes per dimension, the old fit
reported convergence at iteration 92 after accepting a joint step of about
`1.907e-6`; its likelihood change became zero although preceding iterations
still gained about 0.023 each. A flat damped step is not convergence evidence.

Seven-node runs reproduce this control-flow defect only. They are not
study-quality numerical precision or recovery evidence. Repository study
settings require at least 121 nodes per dimension and larger rules until
estimates stabilize; this patch does not claim such a study has been completed.

The previous three-seed exploratory run (1000–1002) yielded mean focal SDs
(1.216, 0.876). Continuing with the parameter-change guard yielded
(1.377, 0.806), against generating SDs (1.25, 0.8). Overall Euclidean error
increased from about 0.083 to 0.127. These runs do **not** demonstrate improved
recovery. The second coordinate alone is not an adequate recovery claim.
The S1 consumer's convergence-dependent gate must remain false when the
stricter estimator does not converge; row-order agreement does not override it.

## Full-step positive-path regression

A separate inherited stopping defect required a history of partial-recovery
progress before permitting convergence. That history was recorded only after
rejecting a joint update; an EM run accepting every joint update could therefore
never report convergence, even when both tolerance checks passed. The history
flag had no role in rollback or stall handling. This follow-up removes only that
prerequisite and its unused bookkeeping; the full-step, finite parameter-change,
absolute likelihood-change, and no-rollback guards remain.

The real-binding positive regression uses the existing correlated-primary
simulation (seed 1000, 300 people), all six items anchored, q7, a 12-sweep budget,
and an explicitly loose tolerance of 0.1. It reconstructs the final parameter
change from the last two accepted prior states. Before the fix, every sweep was
accepted in full and both tolerances passed, but convergence was false. The
fixture tests stopping control flow only, not precision, stationarity of the
inner optimizer, or recovery. The loose fixture tolerance does not change any
estimator default or production tolerance. The permanent regression failed
before the deletion at the convergence assertion, after its full-step and
parameter/likelihood assertions passed. After the deletion it returned
`tolerance_met` at sweep 1, with parameter displacement 0.05371597503918927 and
likelihood change 1.6380025410030612; both satisfy this deliberately loose
control-flow fixture. The full-step positive regression and the CPU/GPU
damped-step negative regressions passed together (3 tests), Rust two-tier tests
passed 24 with 2 ignored, and binding tests passed 9. These are local author-run
checks, not hosted or independent approval.

## Open optimizer question

The existing scale-recovery initial step `scale_alpha = 0.1` was introduced in
commit `1d18b8d26602ac7fcd6450751a6b2baa359168ed`. Its methodological derivation
has not been verified. This patch leaves it unchanged and does not attribute
that constant to Chalmers or mirt. A separate optimizer investigation must
justify or replace the trust-region policy and demonstrate recovery with
adequate quadrature precision; convergence labels must not be relaxed to make
that investigation pass.

## Frozen-support diagnostic contract

The `fixed_*` traces evaluate current item slopes and thresholds on the initial
primary N(0, I) and specific N(0, 1) quadrature supports with their standard GH
weights. Support coordinates and weights are retained once before the fitting
loop; later focal means, covariances and specific SDs do not remap this pass.
Item parameters are not frozen. The production E-step still uses its current
remapped primary and scaled specific supports without any change.

This is a separate implementation diagnostic, not the production likelihood,
prior-update statistic, stopping criterion, or a monotonic EM objective. Its LL
need not increase after a production update. Existing fixture-specific assertions
about improvement are retained, not promoted to a general guarantee. The new
explicit low-Q Rust control distinguishes fixed from remapped support and checks
that changed item parameters still affect the fixed evaluation. In the bounded
scratch proposal that Rust control is uncompiled and unexecuted; source wiring
and synthetic finite-sum controls are not native or scientific acceptance.

## Recovery and stopping measure separation

The frozen standard-normal evaluation is output-only. Mean and scale recovery
compare candidate likelihoods with the current remapped production likelihood
`ll`, including the unchanged `32 * EPSILON * (1 + abs(ll))` improvement margin.
Before the frozen diagnostic was separated, the duplicated diagnostic E-step
used the same current supports, weights and item parameters as that production
E-step. Restoring its baseline to `ll` preserves that measure; it does not
justify or redesign the existing damping or scale-recovery policy.

`loglik_trace` now retains the current production evaluation even when a
convergence check exits before the next M-step. The previous production value
is captured before the append and still supplies the same stopping check.
For a tolerance exit with at least two evaluations, `final_loglik_change` is
`loglik_trace[-1] - loglik_trace[-2]`; the absolute-likelihood bound uses
`loglik_trace[-2]`. Frozen likelihoods do not supply either value. This changes
the returned trace at tolerance exits, not the convergence predicate. A
`prior_update_stalled` exit can occur immediately after an update without a
new checked likelihood evaluation; this trace contract does not certify that
exit or turn its last checked delta into a new convergence result.

The successor proposal has source/scalar/trace-oracle controls only. Its Rust
code and retained numerical regressions still require execution by the existing
native owner; those controls do not establish compiled, GPU or recovery acceptance.

## Reference

Chalmers, R. P. (2012). mirt: A multidimensional item response theory package for
the R environment. *Journal of Statistical Software, 48*(6), 1–29.
https://doi.org/10.18637/jss.v048.i06

The cited paper was read from the existing Zotero item `MTJVTFN3`, PDF attachment
`IUL6W6A6`; a copy is attached as [chalmers-2012-mirt.pdf](papers/chalmers-2012-mirt.pdf).
The [publisher page](https://www.jstatsoft.org/article/view/v048i06) grants
CC BY 3.0 redistribution with attribution. The PDF is unchanged. The two pinned
source links were saved to Zotero as `KP6V3NIH` and `JRM2T2X5`.

## Local regression evidence

On September 30, 2026, the real-binding seed-1000 regression passed for CPU and
GPU requests. The separate hardware receipt confirmed Metal execution on
Apple M5, not CPU fallback. At the 100-iteration budget the Metal fit reported
`converged = false`, `termination_reason = step_limited`, likelihood change
0.020171165466308594, and full-candidate parameter displacement
2.3602703905750633. This is evidence for honest stopping behavior, not recovery
acceptance. The built extension SHA-256 was
`f699af163fe13b444995142ed345554f4c806b705630012db744955c7b38c442`.
