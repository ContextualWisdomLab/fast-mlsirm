# `parallel.rs` naming disambiguation

## Changed

- Documentation only (Issue #2009): the module rustdoc of
  `crates/mlsirm-core/src/parallel.rs` and `ARCHITECTURE.md` §5.1 now state
  that the module implements Horn's parallel analysis (component retention),
  not thread or data parallelism, and point readers of the CPU-parallelism
  track (#2001/#2002/#2003) to the bifactor E-step person sweep
  `bifactor_grm::e_step`. The module is not renamed, so Rust imports and the
  Python `parallel_analysis` API are unchanged.
- `tests/test_parallel_analysis_naming_contract.py` pins the disclaimer, the
  absence of threading primitives in `parallel.rs`, the existence of the named
  E-step person loop, and the matching `ARCHITECTURE.md` entry.
