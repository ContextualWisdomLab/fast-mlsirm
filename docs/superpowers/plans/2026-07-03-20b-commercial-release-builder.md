# 20B Commercial Release Builder Implementation Plan

## Scope

Implement the commercial release builder inside the existing repository. Reuse
the current release evidence scripts and standard library only.

## Steps

1. Add `scripts/build_commercial_release.py`.
   - Accept `--out`, `--dist`, `--contract-value-krw`, `--python`,
     `--require-rust`, `--check-import`, and `--skip-build`.
   - Run dist build, release acceptance, benchmark report, sales readiness,
     buyer packet, release evidence index, and final sales-readiness gate.
   - Stop on the first failed stage and record the failure.
   - Emit `commercial_release_manifest.json` and
     `commercial_release_report.html`.

2. Add orchestrator tests.
   - Success path with a fake runner.
   - Failed-stage stop behavior.
   - Custom `--dist` build output path.

3. Extend the 20B readiness gate.
   - Add the builder to required product evidence files, document tokens, and
     completion checks.
   - Update test fixtures and `product_completion_manifest.json`.

4. Update buyer-facing evidence docs.
   - README commercial workflow.
   - Commercial readiness.
   - Enterprise sales readiness.
   - Release acceptance guide.
   - KRW 2,000,000,000 product readiness.
   - Buyer storyboard, Figma packet, ROI evidence model, and enterprise demo
     README/manifests.

5. Verify locally.
   - `python -m pytest`
   - `cargo test --workspace`
   - `cargo test --manifest-path crates/fast-mlsirm-py/Cargo.toml`
   - `python -m build`
   - `python -m twine check dist/*`
   - `python scripts/build_commercial_release.py --out commercial-release --require-rust --check-import`

6. Ship.
   - Commit on `codex/20b-commercial-release-builder`.
   - Open PR.
   - Inspect live PR/check/ruleset state.
   - Address actionable review or CI failures.
   - Merge the checked head; review delay or queued checks alone are not a
     blocker after fresh local and repository evidence is recorded.
