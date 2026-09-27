# #1976 bifactor GRM EM stall — root cause and fix

## Root cause

At large Gauss–Hermite `q` (Golub–Welsch on-demand rules), some prior
weights underflow to **exact 0**. The E-step then sets `log_wg[g] = -inf`
and builds `gen_log[g] = log_wg[g] + general-only`, which stays `-inf`.

The joint-posterior path computed

```text
others = gen_log[g] - log_wg[g]   # (-inf) - (-inf) = NaN
```

NaN posts were added into expected counts, the M-step saw a non-finite
objective and left parameters at the start (`a_G = 1.0`, `a_S = 0.8`),
while the observed-data loglik (still finite from non-zero-weight nodes)
plateaued enough for the relative-change stop to fire as
`converged=True` / `termination_reason="tolerance_met"`.

Oakes SE at the stalled point correctly reported non-PD; defect is EM-only.

## Fix

1. **E-step (CPU + GPU WGSL):** skip nodes with non-finite prior log-weight
   instead of subtracting; reject non-finite posts before accumulating
   counts (`general_only_without_prior`).
2. **Fail-closed:** if relative change would claim `tolerance_met` but every
   item parameter is still bit-identical to the start, reclassify as
   `converged=False`, `termination_reason="numerical_em_stall"`.
3. **Python contract:** document the new termination reason on single-group
   and multigroup fit results.

## Evidence (MacBookAir, `CARGO_TARGET_DIR=~/late-life-compute/target-bf1976`)

```text
cargo test --manifest-path crates/mlsirm-core/Cargo.toml --no-default-features bifactor
```

- `zero_prior_weight_nodes_do_not_nan_estep_counts` … ok
- `refuse_tolerance_reclassifies_bit_identical_start` … ok
- `dense_quadrature_fit_never_claims_tolerance_at_start_slopes` (q=421) … ok
- lib suite: **30 passed** (bifactor filter); overall cargo exit 0

Local Mac cargo was skipped (lock contention with OLS-HC worker per
coordinator).

## Files

- `crates/mlsirm-core/src/bifactor_grm.rs`
- `crates/mlsirm-core/src/gpu_bifactor.rs`
- `python/fast_mlsirm/bifactor_grm.py`
- `python/fast_mlsirm/bifactor_multigroup.py`
- `tests/unit/bifactor_grm_tests.rs`
- `tests/test_bifactor_grm.py`
