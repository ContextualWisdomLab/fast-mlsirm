# Release cut 0.9.2

## Changed

- Stage package version metadata at `0.9.2` across Python, Rust, `python/fast_mlsirm/_version.py`, and the derived lock metadata on the Draft release branch. This fragment does not claim that a `v0.9.2` tag, package, or immutable GitHub release already exists.
- The historical 2026-08-27 cut folded the then-current 17 unreleased fragments into the draft `[0.9.2]` section. The branch has since been merged forward non-destructively with protected `main`, including the binary-response Measurement contract, so that historical section is no longer complete release provenance for the code that would now be tagged.
- Keep the current authoritative fragments until the exact tag target is ready. Before release, regenerate the managed `Unreleased` block and the dated `0.9.2` release body from that exact inventory, fold every released fragment without losing its evidence, and require `python scripts/render_changelog_fragments.py --check CHANGELOG.md` plus the release-consistency gates to pass on one unchanged head.
