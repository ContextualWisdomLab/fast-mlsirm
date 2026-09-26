# Red Semgrep gate on every PR

## Fixed

- The central `Semgrep (multi-language SAST)` gate reported three blocking
  WARNING findings on `main`, so it failed on every pull request regardless of
  its contents. The org ruleset gates on that workflow passing, so this blocked
  merges repository-wide. Reproduced locally with the same ruleset
  (`semgrep --config=p/default --severity=WARNING --severity=ERROR`), which
  returns the same three.
- `python/fast_mlsirm/dif.py` built its deprecated-alias docstrings by indexing
  `globals()` with loop variables drawn from a literal table three lines above.
  The rule cannot see that the keys are literals, and the indirection bought
  nothing: the loop now names the function objects directly, so a typo fails at
  import instead of at runtime, and the finding disappears with cleaner code.
- `tools/inventory_public_api.py` now enumerates repository-owned Python files
  and parses otherwise-unloaded public modules with `ast` instead of importing
  discovered module names. A regression fixture proves an import-time side
  effect in a discovered module is not executed, while constructor projections
  retain dataclass, enum, protocol, exception, and inherited signatures.

Neither change weakens the gate: both dynamic execution primitives are removed,
with no suppression or rule downgrade.
