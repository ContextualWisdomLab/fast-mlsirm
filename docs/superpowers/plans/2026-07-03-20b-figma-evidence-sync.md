# 20B Figma Evidence Sync Implementation Plan

## Goal

Implement a Figma buyer-evidence sync gate that makes the static Figma design
packet auditable inside the commercial release evidence chain without using
Figma Code Connect.

## Steps

1. Add focused tests.
   - Cover successful `figma_evidence_sync_manifest.json` and
     `figma_evidence_sync_report.html` generation.
   - Cover missing required evidence tokens.
   - Cover `code_connect: true`.
   - Cover commercial release builder stage integration.
   - Cover sales-readiness validation and HTML SHA256 mismatch.

2. Add the builder.
   - Create `scripts/build_figma_evidence_sync.py`.
   - Read `examples/enterprise_demo/figma_design_packet.json`.
   - Optionally read an exported live Figma metadata snapshot.
   - Validate required frame ids, required evidence tokens, Figma design URL,
     and Code Connect disabled status.
   - Emit manifest and HTML report with restrictive CSP and digest metadata.

3. Integrate release gates.
   - Add the Figma sync stage to `scripts/build_commercial_release.py` after PR
     queue governance.
   - Add `--skip-figma-evidence-sync`, `--figma-metadata-snapshot`, and
     `--figma-url`.
   - Add `--figma-evidence-sync` and `--require-figma-evidence-sync` to
     `scripts/sales_readiness.py`.

4. Update buyer evidence.
   - Update README, commercial readiness, enterprise sales readiness, release
     acceptance, 20B product readiness, buyer storyboard, Figma product design
     packet, ROI evidence model, and enterprise demo README.
   - Add `figma_evidence_sync` to `product_completion_manifest.json`.
   - Add design-evidence confidence and required artifacts to
     `roi_manifest.json`.
   - Update `figma_design_packet.json` handoff metadata.

5. Verify.
   - `python -m py_compile scripts/build_figma_evidence_sync.py scripts/build_commercial_release.py scripts/sales_readiness.py`
   - `python -m pytest tests/test_figma_evidence_sync.py tests/test_commercial_release_builder.py tests/test_sales_readiness.py`
   - `python -m pytest`
   - `cargo test --workspace`
   - `cargo test --manifest-path crates/fast-mlsirm-py/Cargo.toml`
   - `python -m build`
   - `python -m twine check dist/*`
   - `python scripts/build_figma_evidence_sync.py --out figma-evidence-sync-live`
   - `python scripts/build_commercial_release.py --out commercial-release-figma-sync --require-rust --check-import --skip-build`

6. Ship.
   - Commit on `codex/20b-figma-evidence-sync`.
   - Push branch and create a PR.
   - Inspect PR checks, review threads, mergeability, and rulesets.
   - Merge when current-head evidence is sufficient. Review process delay or
     queued checks alone are not blockers.
