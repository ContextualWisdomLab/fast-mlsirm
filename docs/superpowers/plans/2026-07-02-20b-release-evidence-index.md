# 20B Release Evidence Index Implementation Plan

## Scope

Implement the release evidence index inside the existing repo and package
support scripts. Keep the work stdlib-only and avoid a new library, submodule,
or hosted service.

## Steps

1. Add `scripts/build_release_evidence_index.py`.
   - Read dist artifacts, acceptance, benchmark, sales readiness, and buyer
     packet manifests.
   - Resolve linked HTML/ZIP artifacts.
   - Calculate SHA256 digests and coverage booleans.
   - Emit `release_evidence_index.json` and `release_evidence_index.html`.

2. Extend `scripts/sales_readiness.py`.
   - Add `--release-evidence-index`.
   - Add `--require-release-evidence-index`.
   - Validate status, contract value, required coverage, dist hashes, failure
     list, HTML report existence, and HTML report SHA256.

3. Extend `scripts/build_buyer_packet.py`.
   - Add optional `--release-evidence-index`.
   - Include release index JSON/HTML under `release/` when supplied.
   - Keep it optional to avoid recursive packet/index hashing.

4. Update evidence docs and manifests.
   - README commercial workflow.
   - Commercial readiness, enterprise sales readiness, release acceptance, 20B
     product readiness, storyboard, Figma packet, ROI evidence model.
   - Product completion and Figma design packet manifests.

5. Add tests.
   - Release index success and digest-mismatch failure.
   - Sales readiness release-index validation success and SHA failure.
   - Buyer packet optional release-index inclusion.

6. Verify.
   - `python -m pytest`
   - `cargo test --workspace`
   - `cargo test --manifest-path crates/fast-mlsirm-py/Cargo.toml`
   - `python -m build`
   - `python -m twine check dist/*`
   - release acceptance, benchmark report, buyer packet, release index, and
     final sales readiness gate.

7. Ship.
   - Open PR from `codex/20b-release-evidence-index`.
   - Address review/CI findings when actionable.
   - Merge after the checked head is proven; reviewer delay alone is not a
     code blocker.
