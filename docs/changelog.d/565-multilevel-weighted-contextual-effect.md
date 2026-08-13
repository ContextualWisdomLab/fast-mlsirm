# Rust-owned sparse weighted contextual-effects predictor

## Added

- Added `mlsirm_core::multilevel::weighted_contextual_effect`: the contextual term `sum_h w_ph u_h` of the multilevel linear predictor (Browne, Goldstein, & Rasbash, 2001) over a sparse CSR-style cross-classified multiple-membership design. Ordinary nesting is the one-hot special case (`w_ph = 1` for exactly one edge per dimension), not a separate code path.
- Deterministic regardless of edge order within an observation, observation order, or worker count: each row is summed in ascending context-index order and rows are independent, backed by a bounded manual `std::thread::scope` worker pool (no new dependency).
- Added one-hot-nesting-parity, weighted-membership, cross-classified-dimension, permutation-invariance (edge order and row order), and worker-count-determinism Rust unit tests, plus fail-closed validation of malformed CSR offsets, out-of-range context indices, and non-finite/negative weights.
- Added a `_multilevel_core` PyO3 extension module (dual-`PyInit_*` pattern, matching bifactor/rotation/rating-range) exposing `weighted_contextual_effect` as a marshal-only numpy binding, plus `fast_mlsirm.multilevel.weighted_contextual_effect`, which marshals a validated `ContextMembershipDesign` and a per-context effect mapping into the Rust call and back.
- Reserves the Bayesian/MCMC estimation of the random effects `u_h` themselves, longitudinal state transitions, uncertainty, GPU batch path, and fairness/DIF work for the later staged PRs in issue #565.
