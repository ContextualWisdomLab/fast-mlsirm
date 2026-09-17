# Graphify tooling investigation for #1847 and #1833

## Fixed

- #1847: `to_json`'s node-count shrink guard refused a `cluster-only` write
  on an unchanged graph after `build_from_json`'s ghost-merge pass
  legitimately collapsed a manifest-derived duplicate node into its
  AST-canonical twin (`crate:mlsirm-core` / `pkg_mlsirm_core`, zero
  incident edges dropped). `build_from_json` now records the collapsed
  count (`_ghost_dedup_count`); the shrink guard excuses a drop only when
  fully explained by it. Upstream PR:
  https://github.com/Graphify-Labs/graphify/pull/3623.
- #1833: a workspace-only `Cargo.toml` (`[workspace]`, no `[package]`)
  correctly emits no package node, but the extractor's zero-node detector
  could not tell that apart from an unexplained failure and printed a
  persistent warning every run. The manifest parser now marks this case
  `skipped`, so the by-design exclusion is explicit instead of warning.
  Upstream PR: https://github.com/Graphify-Labs/graphify/pull/3622.
- No fast-mlsirm runtime, Cargo, or Python code changed — both issues were
  tooling-only (Graphify artifact refresh/reviewability), confirmed via a
  RED-then-GREEN regression test in the `seonghobae/graphify` fork before
  the upstream PRs were opened.
- Pinned install/rollback instructions for trying the fork fix locally are
  recorded on fast-mlsirm#1833.
