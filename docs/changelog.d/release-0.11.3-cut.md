# Release cut 0.11.3

## Changed

- Project version is bumped to 0.11.3 in `pyproject.toml`, `crates/mlsirm-core`,
  and `crates/fast-mlsirm-py`. The accumulated `Unreleased` notes now form the
  `[0.11.3] - 2026-09-18` release section, headlined by the bifactor GPU E-step
  Metal/WebGPU workgroup-dimension split (#1987): dispatches are factored across
  `(x, y, z)` from runtime adapter limits, WGSL uses an f32-representable
  zero-mass sentinel, and the PyO3 cdylib again enables the `gpu` default feature.
- This cut removes the standing predecessor note `release-0.11.2-cut.md`, whose
  substance is permanently recorded in the `[0.11.2] - 2026-09-18` section and
  in git history.
- Released authoritative fragments are removed from `docs/changelog.d`; the
  directory again holds only genuinely unreleased notes.
