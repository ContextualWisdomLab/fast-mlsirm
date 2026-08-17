# Doctoring record: dedicated GRM 500-rep recovery lane

## Decision

The multidimensional Graded Response Model Monte Carlo study
`mlsirm-core/lib/mlsirm_core::grm::tests::mc_grm_recovery_500` is scientific
acceptance evidence, not a bounded pull-request sentinel. It runs 500
replications across two- and three-dimensional designs under normal and
standardized right-skew traits, then gates loading bias, loading RMSE,
threshold RMSE, theta correlation, and convergence.

That study finished its first two cells inside the generic ignored-shard
1,800-second subprocess budget and then died at the process deadline. Expanding
the generic `STATISTICAL_TEST` timeout would raise the blast radius for every
ordinary ignored study. Reducing replications would destroy the required
recovery evidence.

The operating policy is therefore:

1. exclude the test from the 12-way ignored-shard inventory by its
   target-qualified Cargo identity;
2. execute it exactly once in a dedicated `grm-recovery` job with a 120-minute
   outer ceiling, `--ignored --exact --nocapture --test-threads=1`, and
   `persist-credentials: false`;
3. keep the job on `workflow_dispatch`, the `17 2 * * *` cron, and `v*` tags
   only — never on pull-request CI;
4. publish `grm-recovery-study.log` as a 90-day Actions artifact so a buyer or
   reviewer can read the printed bias/RMSE/convergence/theta-correlation lines
   after the job log expires.

The scientific assertions inside the Rust test do not change. This record
governs orchestration, credential hygiene, and evidence retention only.

## Operator next action

After this lane lands on the default branch, dispatch **Statistical Studies**
once and download the `grm-recovery-study-<run_id>` artifact. Confirm all four
`[grm MC D=…]` lines are present and that the job finished under 120 minutes on
a cold `ubuntu-latest` runner. If the artifact is missing, treat the run as
failed even when `cargo test` printed locally — the contract requires
`if-no-files-found: error`.

## Rollback

Restoring the test to the generic shard reintroduces the 1,800-second
process-level timeout that killed a completed D=2 pair. A safer rollback is to
keep the dedicated job and revert only a failed artifact or credential change.

## References

GitHub, Inc. (2026). *Storing and sharing data from a workflow*. GitHub Docs.
https://docs.github.com/en/actions/using-workflows/storing-workflow-data-as-artifacts

Samejima, F. (1969). Estimation of latent ability using a response pattern of
graded scores. *Psychometrika Monograph Supplement, 17*.

Svetina, D., Valdivia, A., Underhill, S., Dai, S., & Wang, X. (2017). Parameter
recovery in multidimensional item response theory models under complexity and
nonnormality. *Applied Psychological Measurement, 41*(7), 530–544.
https://doi.org/10.1177/0146621617703180
