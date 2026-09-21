# E G+4+W FIPC Mean Drift Diagnosis

## Scope

This diagnosis covers the focal-prior mean path only. It does not assess row
order, AC SlopePrior, or scorer changes. The source consumer log is the
artifact at `/data/orca/workspaces/fmls-g4w-fipc-s1-verify-run/logs/consumer_32807ed0.log`.

## Artifact identity finding

The requested worktree tip is:

```text
32807ed0 fix(two-tier): accept full remapped-LL mean steps earlier
```

The consumer JSON reports both `sha` and `build_source_sha` as:

```text
6c30712db3a318211625db29eb6480ecab393c89
```

The loaded extension is `/data/orca/workspaces/fmls-perf-diag-venv/lib/python3.12/site-packages/fast_mlsirm/_core.cpython-312-x86_64-linux-gnu.so`, with SHA-256
`a0453b7af30038e5f848c7a1f0735c683e41fa809341770c59e123a2d10978ed`.
Consequently, this run is not evidence from tip `32807ed0`; it loaded the
earlier `6c30712d` build. Rebuild/reinstall the extension from `32807ed0` and
rerun the same consumer before attributing behavior to the mean-first patch.

## What the stale-build run does show

- `n_accepted_prior_steps=100`, `n_rollback_full=0`, and
  `consecutive_rollback=0`.
- `termination_reason=max_iter_reached` and `converged=false`; this is an
  iteration-budget failure, not a rollback stall.
- `fixed_loglik_trace` and `loglik_trace` are identical for all 101 entries.
  Every one of the 100 deltas is positive: the first is `5.87969249178218`
  and the last is `0.0256721444957293`.
- The focal mean moves from `[0.24755630586372657,
  -0.023544727718702082]` to `[0.39300025895046864,
  -1.4203648508204998]`.
- The final covariance is `[0.9026153511786992, 0.025924181617726673,
  0.025924181617726673, 0.9136998271044843]`, and the final specific SD is
  `[1.0, 1.0]`.
- `prior_mean_trace` has 200 values (100 updates), but
  `prior_update_decision_trace` is absent/empty in the serialized consumer
  result. Thus the run cannot identify whether the current mean-first branch
  accepted a full step, used backtracking, or used a fallback branch.

## Diagnosis

The observed mean drift is real in the stale-build run, but it is not
classified as overshoot. The available evidence shows a monotonically
improving fixed-measure objective, no full rollbacks, and a nonzero but
shrinking final LL increment. No objective/gradient/identification/scale
evidence demonstrates an overshoot. The means appear to be continuing toward
the focal posterior moment while the run is simply capped at 100 iterations;
the consumer gate therefore correctly fails convergence, although the
numerical path itself does not show an objective regression.

The stronger release-blocking finding is provenance: the consumer tested
`6c30712d`, not `32807ed0`. The current tip's mean-first acceptance behavior
must be evaluated only after rebuilding the extension and confirming the
reported source SHA equals `32807ed0`. Do not add a mean optimizer patch from
this stale run, and do not use the recovery-test fixture as an optimizer
target.

## Required rerun

1. Build/install the Python extension from `32807ed0`.
2. Confirm the consumer reports `sha=32807ed0` and a changed extension hash
   where expected.
3. Rerun the exact s1 consumer configuration (`n_persons=180`, `n_items=6`,
   `n_primary=2`, `n_specific=2`, `q_primary=q_specific=7`, `max_iter=100`,
   `tol=1e-5`, `estimate_specific_vars=false`).
4. Preserve `prior_update_decision_trace`, objective deltas, fixed-measure
   deltas, covariance/scale traces, and any gradient or identification
   diagnostics in the resulting JSON.
