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
- `tools/inventory_public_api.py` calls `importlib.import_module(modname)` with
  a name from `pkgutil.walk_packages(fast_mlsirm.__path__, ...)`. The only
  importable values are this package's own installed submodules, there is no
  caller-supplied input, and the file is a repository tool that never ships in
  the wheel, so it carries a scoped `# nosemgrep` with that justification rather
  than a refactor.

Neither change weakens the gate: the suppression is per-rule, per-line, and
recorded separately from the blocking count by the central workflow.
