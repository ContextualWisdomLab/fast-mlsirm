# PR #2043 L3 contract verification

Verified against rebased head `756ba5dc` (PR head `f5839e42` before the
rebase onto `origin/main` at `99c228a8`).

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
- **Provenance: incomplete at the public Python result.** Rust stores
  `effective_device` and `estep_shards` in `BifactorGrmResult`
  (`crates/mlsirm-core/src/bifactor_grm.rs:207-235`), and PyO3 inserts both
  keys into its dict (`crates/fast-mlsirm-py/src/lib.rs:1469-1482`). However,
  `BifactorGrmFit` declares no matching fields
  (`python/fast_mlsirm/bifactor_grm.py:131-162`), and its construction ends at
  `n_parameters` without reading either key (`:311-331`). Therefore the public
  Python API drops the provenance required by the L3 contract.
- **Scope controls: verified.** The PR diff adds no `two_tier` file and no new
  `atol`/`rtol` assertion. Generic exhaustive matches treat `Device::Split` as
  CPU outside the single-group bifactor split executor; no model formula was
  changed outside that path.

This closes only the #2001 §3 same-host L3 split pilot, subject to the public
Python provenance gap above. It does not implement or close L4 remote or
distributed execution, including Valkey/Streams work tracked by #2001, #2039,
and #2048.

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
