# 20B Procurement Due-Diligence Implementation Plan

## Scope

Implement procurement due-diligence evidence inside the existing repository.
Use only Python standard library APIs and the existing release evidence
pipeline.

## Steps

1. Add `scripts/build_procurement_due_diligence.py`.
   - Accept `--repo-root`, `--dist`, `--commercial-release-manifest`, `--out`,
     `--repo`, `--contract-value-krw`, and `--offline-github`.
   - Parse wheel metadata, source distribution metadata, project metadata,
     policy files, commercial release manifest, and GitHub snapshot state.
   - Emit `procurement_due_diligence_manifest.json` and
     `procurement_due_diligence_report.html`.

2. Integrate the commercial release builder.
   - Run procurement due diligence by default after the commercial release
     manifest and final sales-readiness evidence exist.
   - Keep `--skip-procurement-due-diligence` for deterministic partial runs.
   - Support `--offline-github` for local smoke verification.

3. Extend the sales-readiness gate.
   - Add `--procurement-due-diligence`.
   - Add `--require-procurement-due-diligence`.
   - Validate status, contract value, failed checks, category coverage, HTML
     report existence, and HTML report SHA256.

4. Update buyer-facing evidence docs and manifests.
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
   - `python scripts/build_commercial_release.py --out commercial-release --require-rust --check-import --offline-github`
   - `python scripts/sales_readiness.py ... --require-procurement-due-diligence`

6. Ship.
   - Commit on `codex/20b-procurement-due-diligence`.
   - Open PR.
   - Inspect live PR/check/ruleset state.
   - Address actionable review or CI failures.
   - Merge the checked head; review delay or queued checks alone are not a
     blocker after fresh local and repository evidence is recorded.
