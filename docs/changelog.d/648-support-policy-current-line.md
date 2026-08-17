# Current support-policy version line

## Changed

- Aligned the public security and support policies with the released `0.7.x` pre-1.0 package line instead of the obsolete `0.1.x` policy.
- Reframed support around released, documented public API and packaging behavior, preserved conservative high-stakes/certification/SLA boundaries, and clarified Rust-first production numerical ownership versus explicit reference/parity paths.
- Added a repository contract that derives the supported minor line from `pyproject.toml` so future package-version changes cannot silently leave `SECURITY.md` or `SUPPORT.md` stale.
