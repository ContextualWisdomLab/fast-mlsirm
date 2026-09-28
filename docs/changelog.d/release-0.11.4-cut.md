# Release cut 0.11.4

## Changed

- Project version is bumped to 0.11.4 in `pyproject.toml`, `crates/mlsirm-core`,
  and `crates/fast-mlsirm-py`. The accumulated `Unreleased` notes now form the
  `[0.11.4] - 2026-09-18` release section, headlined by the PyPI package-description
  boundary repair (#1993): `README.md` no longer carries internal commercial-boundary
  vocabulary or repo-relative links that 404 on the registry page, so the corrected
  immutable description can ship after the 0.11.3 page. The section also folds the
  two-tier / multi-primary Oakes SE and streaming E-step memory work (#1992) and the
  support-policy / bifactor quadrature test repairs that cleared red `main`.
- This cut removes the standing predecessor note `release-0.11.3-cut.md`, whose
  substance is permanently recorded in the `[0.11.3] - 2026-09-18` section and
  in git history.
- Released authoritative fragments are removed from `docs/changelog.d`; the
  directory again holds only genuinely unreleased notes.
