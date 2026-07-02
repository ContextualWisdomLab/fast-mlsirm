# KRW 2,000,000,000 Buyer Evidence Packet Design

## Objective

Make the existing `fast-mlsirm` 20B readiness evidence easier for a buyer to
review by generating one portable packet that contains release artifacts,
acceptance output, sales-readiness output, product docs, demo manifests, and
SHA256 digests.

## Boundaries

- Do not add a separate library, submodule, hosted app, artifact registry, or
  signing service.
- Do not use Figma Code Connect.
- Do not change formulas, estimators, or diagnostics semantics.
- Use only Python standard library modules for packet generation.

## Product Design

The existing Figma file
`https://www.figma.com/design/qD34PfMH8Kr41tFdqLCkem` remains the visual
reference. The `06-procurement-packet` screen maps to the new packet outputs:
`buyer_evidence_manifest.json` and
`fast_mlsirm_buyer_evidence_packet.zip`.

## Data Analytics

The packet manifest records:

- `contract_value_krw`;
- `generated_at`;
- `source_commit`;
- `artifact_count`;
- required evidence coverage booleans;
- per-file SHA256 digests and byte sizes;
- packet zip path and packet zip SHA256.

This is procurement evidence, not a valuation guarantee or customer-specific
ROI calculation.

## Architecture

Add `scripts/build_buyer_packet.py` as a narrow command-line utility. It reads
an existing `acceptance_summary.json`, an existing
`sales_readiness_manifest.json`, the built `dist/` artifacts, and repo-local
docs/manifests. It writes a manifest plus zip file. Extend
`scripts/sales_readiness.py` with optional packet validation flags so existing
default gates keep working.

## Test Strategy

- Unit-test packet creation with synthetic acceptance, sales, dist, docs, and
  demo manifests.
- Unit-test packet validation success and SHA mismatch failure in
  `sales_readiness.py`.
- Run focused tests, full Python tests, Rust tests, package build, release
  acceptance, packet build, and 20B readiness gate with packet validation.
