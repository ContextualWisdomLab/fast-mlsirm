# Public API naming convention and unsourced-defaults policy (#1959)

## Added

- **ADR-0028** (`docs/adr/0028-public-api-naming-and-defaults-policy.md`,
  Proposed): one verb-first naming convention and one unsourced-defaults
  policy (numerical-precision controls, decision thresholds, seeds, model
  choice) for every public `fast_mlsirm` callable and PyO3 entry point.
- `tools/inventory_public_api.py`: regenerates a full public-callable
  inventory (`docs/api/inventory-YYYYMMDD.csv`) via static analysis, no Rust
  build required.
- `tools/classify_renames_and_defaults.py`: mechanically applies ADR-0028's
  rules to the inventory, producing
  `docs/api/renames-and-defaults-YYYYMMDD.csv` with a proposed name and a
  per-default decision (`keep+source` / `require` / `change`) for every
  callable, absorbing #1958/#1960's completed `dif_polytomous*` outcome.

No code, name, or default changes in this PR (Phase 1, documentation only);
per-module implementation is tracked in the sub-issues this PR opens.
