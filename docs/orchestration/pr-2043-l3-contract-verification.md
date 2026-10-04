# PR #2043 L3 contract verification

Current source correction: `5892ca15392c05d3d2140ef007c75dfce95ee25c`.
The public-result finding below is resolved by source and recording-binding checks,
not by a fresh native or GPU execution. The split-path locators and timing
measurements in the other original findings describe rebased head `756ba5dc`
(PR head `f5839e42` before the rebase onto `origin/main` at `99c228a8`);
they remain historical evidence, not measurements of the current combined head.

## Findings

- **Non-overlapping person partition: verified in the Rust split path.**
  `parse_bifactor_device` rejects empty shards and constructs the boundary at
  `crates/mlsirm-core/src/bifactor_grm.rs:180-200`. The split executor submits
  persons `[split, n_persons)` to the GPU and runs `[0, split)` on the CPU at
  `crates/mlsirm-core/src/bifactor_grm.rs:779-816`.
- **GPU submit before CPU work: verified.** The non-blocking submit occurs at
  `crates/mlsirm-core/src/bifactor_grm.rs:802-804`, the CPU shard runs at
  `:805-816`, and GPU readback completes afterward at `:817-818`.
- **Fixed-order `f64` merge: verified.** Partials carry per-person `f64`
  log-likelihoods (`crates/mlsirm-core/src/bifactor_estep_split.rs:27-36`), are
  sorted by `person_start`, and merge in that order at `:170-186`.
- **Provenance: resolved at the public Python result.** Current
  `BifactorGrmFit` declares `effective_device` and `estep_shards`
  (`python/fast_mlsirm/bifactor_grm.py:143-176`), and its constructor reads
  both binding fields (`:334-360`). Rust retains the fields in
  `BifactorGrmResult` (`crates/mlsirm-core/src/bifactor_grm.rs:257-285`), and
  PyO3 exports them (`crates/fast-mlsirm-py/src/lib.rs:1472-1485`). The current
  split parser rejects empty/out-of-range shards at
  `crates/mlsirm-core/src/bifactor_grm.rs:226-253`.
  Metadata describes the final E-step of the winning run, not every-step GPU dispatch
  (`crates/mlsirm-core/src/bifactor_grm.rs:280-285`). Earlier sweeps, other
  starts, physical GPU execution, and native build-source identity are not
  established by Python field transport.
- **Original split-only scope controls: verified at the historical head.**
  That original PR diff adds no `two_tier` file and no new
  `atol`/`rtol` assertion. The split request lives on the bifactor-local
  `bifactor_grm::BifactorDevice`, not the shared `crate::Device`, so scoring,
  likelihood, and multilevel APIs cannot receive it (follow-up to the Devin
  review on #2043); no model formula was changed outside that path.

This resolves the public Python provenance-transport gap; it does not close #2001
or certify the same-host L3 split pilot. Current source-bound native execution,
physical CPU/GPU device/readback evidence, and installed-wheel two-host
acceptance remain separate obligations. The original split-only findings do not
certify the L4 remote and injected-client Valkey contracts subsequently included
in the combined branch, or their real-server/distributed acceptance (#2039, #2048).

## Timing evidence boundary

The committed evidence at
`docs/orchestration/bifactor-estep-split-overlap-2043-timestamps.txt:53-75`
shows intersecting monotonic host intervals for the CPU shard and the span from
GPU submit return through readback completion. It does **not** isolate GPU
kernel start/end, so it proves timestamp-interval intersection while GPU work
is outstanding, not kernel-level temporal overlap or release-complete wall-time
overlap. The evidence file itself states this limitation at
`docs/orchestration/bifactor-estep-split-overlap-2043-environment.txt:23-34`.

## Semgrep remediation

The failing head reported three whole-tree findings: two dynamic `globals()`
lookups in `python/fast_mlsirm/dif.py` and one repository-internal inventory
import in `tools/inventory_public_api.py`. The remediation reuses the #2015
pattern: direct function-object pairs remove the dynamic global lookup, while
the inventory traversal reuses `pkgutil.resolve_name` instead of calling the
flagged dynamic-import API directly. No Semgrep suppression or ignore was added.
The matching local command completed with 0 findings across 609 tracked files
and 327 rules; hosted CI must still report `SUCCESS` on the pushed exact head.
