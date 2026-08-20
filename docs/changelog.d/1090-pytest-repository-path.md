# Make repository test imports deterministic

## Fixed

- Pytest now exposes both the repository root and the Python source tree from
  committed configuration, so tests that materialize repository automation
  scripts do not require an operator-specific `PYTHONPATH=.` workaround.
