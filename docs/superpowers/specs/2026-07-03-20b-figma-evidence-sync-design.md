# 20B Figma Evidence Sync Design

## Objective

Close the gap between the static Figma buyer-review file and the repo-local
commercial release evidence. A KRW 2,000,000,000 buyer packet should not rely on
visual design claims alone; it should show that the Figma procurement frame
still references the same buyer packet, release evidence index, procurement
due-diligence, and PR queue governance artifacts that the release gate checks.

## Scope

Add `scripts/build_figma_evidence_sync.py` as a stdlib-only evidence builder in
this repository. A separate library, submodule, hosted design service, or Figma
Code Connect integration is out of scope. The evidence is repository-local and
can optionally consume an exported live Figma metadata snapshot when connector
access is available.

## Product Design Contract

- Keep the existing static Figma design file as the visual buyer-review
  artifact.
- Keep Figma Code Connect disabled.
- Require the procurement packet frame to reference buyer packet, release
  evidence index, procurement due-diligence, and PR queue governance evidence.
- Produce a human-readable HTML report that procurement can inspect without
  opening raw JSON first.

## Data Analytics Contract

The report is not a valuation claim. It is evidence for the `design_evidence`
driver in the ROI model:

- required frame coverage;
- required evidence token coverage;
- Code Connect disabled status;
- optional live metadata snapshot status;
- SHA256 digest for the HTML review report.

## Generated Artifacts

- `figma_evidence_sync_manifest.json` for machine validation;
- `figma_evidence_sync_report.html` for human procurement review.

## Acceptance

The feature is accepted when:

- the builder passes when the design packet includes all required frames and
  tokens;
- the builder fails when required buyer-evidence tokens are missing;
- the builder fails when Code Connect is enabled;
- `scripts/build_commercial_release.py` runs the Figma sync stage by default
  after PR queue governance;
- `scripts/sales_readiness.py --require-figma-evidence-sync` validates status,
  contract value, Code Connect disabled status, category coverage, frame/token
  coverage, HTML existence, and HTML SHA256;
- docs, JSON manifests, Product Design packet, ROI model, and buyer storyboard
  reference the generated artifacts.
