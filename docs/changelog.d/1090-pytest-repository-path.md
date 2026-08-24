# Make repository test imports deterministic

## Fixed

- Pytest now exposes both the repository root and the Python source tree from
  committed configuration, so tests that materialize repository automation
  scripts do not require an operator-specific `PYTHONPATH=.` workaround.
- Agent guidance now derives its advertised Python support floor from the same
  `pyproject.toml` requirement guarded by repository tests, preventing stale
  lower-version setup instructions from diverging from package metadata.
