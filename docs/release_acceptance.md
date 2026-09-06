# Release Acceptance Guide

## Purpose

`fast-mlsirm` is treated as commercially supportable only after this
release-acceptance smoke test passes on the exact release artifact or installed
package.

The script verifies:

- simulation output generation
- fitting with resolved backend (`--backend auto`)
- explicit Rust fit backend when requested
- fit and dimensionality diagnostics output
- diagnostics report HTML rendering
- per-step and total runtime evidence for acquisition/commercial readiness review

## How to Run

```bash
python scripts/release_acceptance.py \
  --out release_acceptance \
  --persons 12 \
  --dims 1 \
  --items-per-dim 2 \
  --latent-dim 1 \
  --max-iter 1 \
  --n-restarts 1 \
  --latent-dims 1,2 \
  --folds 2 \
  --require-rust
```

### Expected success shape

- Exit code `0`
- JSON result printed to stdout with `"status": "ok"`
- `acceptance_summary.json` written under `--out` with step outputs
- each step in `acceptance_summary.json` includes `duration_seconds`
- `acceptance_summary.json` includes `total_duration_seconds`
- Generated artifacts:
  - `simulate/responses.npy`
  - `fit_auto/fit_summary.json`
  - `fit_rust/fit_summary.json` (present when `--require-rust`)
  - `diagnostics_fit/fit_diagnostics.json`
  - `diagnostics_dimensions/dimension_diagnostics.json`
  - `fit_report.html`
  - `dimension_report.html`

## Acquisition / Commercial Readiness Gate

The complete generic buyer-review path is the price-neutral acquisition release
orchestrator:

```bash
python scripts/build_acquisition_release.py \
  --out acquisition-release \
  --require-rust \
  --check-import
```

Unless `--contract-value-krw` is supplied explicitly, the orchestrator leaves
transaction value unset. It builds or reuses the release artifacts, runs release
acceptance, creates benchmark, buyer-packet, release-index, procurement,
PR-queue, and Figma evidence, then finishes with the canonical
`sales_readiness.py --require-acquisition-readiness` gate. The final generic
gate therefore evaluates evidence completeness rather than a hard-coded deal
value.

The canonical acquisition path seals one exact repository revision and carries
that lowercase full `source_commit` through release acceptance and every
sales-readiness manifest. Buyer-packet collection requires the sales-readiness
identity to be present and equal to the sealed packet revision, while the
acquisition stage verifier independently checks the in-memory and persisted
manifest against the same source. A direct legacy/non-Git `sales_readiness.py`
recheck can still inherit an acceptance summary with no source identity and emit
`source_commit: null`; that output is verification-only legacy evidence and is
not admissible to the canonical source-bound buyer packet.

The primary machine-readable outputs are:

- `acquisition-release/acquisition_release_manifest.json` — exact-source bundle
  inventory and SHA-256 evidence digests;
- `acquisition-release/release-acceptance/final_acquisition_readiness_manifest.json`
  — the complete price-neutral readiness decision;
- the bounded benchmark, buyer-packet, release-index, procurement, PR-queue, and
  Figma manifests under the same output tree.

A candidate is ready for buyer review only when the final acquisition-readiness
manifest is `ok`. That means the configured evidence profile is complete and
internally consistent; it does **not** prove a valuation, transaction price,
regulated-use suitability, customer outcome, deployment, or legal transfer.

### Re-check an existing bundle

To validate already-built evidence without rebuilding it, call the canonical
gate with every required acquisition artifact explicitly:

```bash
python scripts/sales_readiness.py \
  --acceptance acquisition-release/release-acceptance/acceptance_summary.json \
  --dist dist \
  --benchmark-report acquisition-release/release-acceptance/benchmark/benchmark_report.json \
  --buyer-packet-manifest acquisition-release/buyer-evidence-packet/buyer_evidence_manifest.json \
  --release-evidence-index acquisition-release/release-evidence-index/release_evidence_index.json \
  --procurement-due-diligence acquisition-release/procurement-due-diligence/procurement_due_diligence_manifest.json \
  --pr-queue-governance acquisition-release/pr-queue-governance/pr_queue_governance_manifest.json \
  --figma-evidence-sync acquisition-release/figma-evidence-sync/figma_evidence_sync_manifest.json \
  --require-rust \
  --require-acquisition-readiness \
  --check-import \
  --out acquisition-release/release-acceptance/recheck_acquisition_readiness_manifest.json
```

The gate intentionally requires all buyer-facing evidence paths when
`--require-acquisition-readiness` is active. Missing evidence fails closed rather
than being silently treated as skipped.

### Legacy compatibility

`scripts/build_commercial_release.py` is retained on this revision for the
historical 20B compatibility profile. It still carries legacy deal-value and
`--require-20b-product` semantics and is **not** the generic product-quality or
acquisition-readiness entry point. Current documentation and the root README use
`scripts/build_acquisition_release.py` instead.

Older automation may still name `sales_readiness_manifest.json` and
`commercial_release_manifest.json`. Those filenames belong to the retained
legacy compatibility flow; they are not substitutes for the generic
`final_acquisition_readiness_manifest.json` decision.

If a real transaction scenario is being evaluated, pass its value explicitly to
`build_acquisition_release.py --contract-value-krw ...`. The scenario is
recorded separately from the generic readiness decision and does not become a
product valuation claim.

## Required Rust Core

`--backend auto` requires the compiled Rust core and fails closed when that
extension is unavailable. Omitting `--require-rust` skips only the second,
explicit `--backend rust` fit; it does not enable a NumPy fallback for the
automatic production acceptance path. Explicit NumPy remains a reference and
parity choice outside this release-acceptance path.
