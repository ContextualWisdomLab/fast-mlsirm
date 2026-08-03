# Changelog render-parity gate and 0.4.0 note restoration

## Fixed

- The `[0.4.0]` CHANGELOG section again carries the binary bifactor
  pilot-calibration handoff, the vectorized NumPy MMLE fallback M-step, and
  the accessible hero-metadata notes. Their authoritative fragments had been
  merged without re-rendering the managed Unreleased block, so the 0.4.0
  release cut inherited a stale block and the published immutable v0.4.0
  GitHub release body silently omitted them; `CHANGELOG.md` is the
  authoritative release record for v0.4.0.
- A repository-level regression test now runs the fragment renderer's check
  against the live `CHANGELOG.md` and `docs/changelog.d` tree, so a pull
  request that adds or edits a fragment without rendering it fails CI before
  any release cut can inherit a stale managed block.
