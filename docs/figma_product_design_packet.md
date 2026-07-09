# Figma Product Design Packet

## Scope

This packet describes the Figma deliverable for the KRW 2,000,000,000 product
readiness gate. Figma Code Connect is explicitly out of scope.

## Figma Build Rules

- Use a Design file, not FigJam or Slides, for product screens.
- Do not use Figma Code Connect.
- Prefer existing design-system components if a buyer or team file provides
  them.
- If no design system exists, use simple auto-layout frames, readable
  typography, and neutral report surfaces.
- Keep screens static unless an interactive hosted prototype is explicitly
  requested later.
- Use the product font discovered from source or design context. If no product
  font exists, use a conservative sans-serif system style.

## Frames

The Figma board should include one frame per storyboard screen:

1. Package Evidence
2. Synthetic Demo Run and Benchmark Evidence
3. Fit Diagnostics
4. Dimensionality Review
5. Report Export
6. Procurement Packet
7. IRT Stability Review
8. Fixed-Item Calibration
9. aFIPC Calibration IA and Screen Definition
10. Fixed-Item Calibration Wireframe

Each frame should include:

- a concise buyer task title;
- the artifact being inspected;
- the go/no-go signal;
- the source file or command that produces the artifact.

The Synthetic Demo Run frame should point to `benchmark_report.json` and
`benchmark_report.html` when showing runtime-budget and command-duration
evidence.

The Procurement Packet frame should include the portable packet outputs:
`buyer_evidence_manifest.json` and
`fast_mlsirm_buyer_evidence_packet.zip`, plus the SHA256 digest status recorded
by the manifest. It should also point to `buyer_evidence_report.html` as the
human-readable review surface for the same evidence. When a benchmark report is
part of the offer, the frame should also show that
`benchmark/benchmark_report.json` and `benchmark/benchmark_report.html` are in
the packet. The same frame should point to `release_evidence_index.json` and
`release_evidence_index.html` as the release-level digest map over the wheel,
source distribution, acceptance output, benchmark evidence, sales-readiness
manifest, and buyer packet. The final buyer-review state should also point to
`commercial_release_manifest.json` and `commercial_release_report.html` as the
single-command stage summary and human review surface, then to
`procurement_due_diligence_manifest.json` and
`procurement_due_diligence_report.html` as the package, policy,
commercial-release, GitHub snapshot, and SHA256 report review surface. It
should also point to `pr_queue_governance_manifest.json` and
`pr_queue_governance_report.html` as the open PR review-state, stale/change
request, release-scope conflict, and SHA256 report review surface. Finally, it
should point to `figma_evidence_sync_manifest.json` and
`figma_evidence_sync_report.html` as the machine and human review surfaces that
verify this static Figma packet still mentions buyer packet, release evidence
index, procurement due diligence, and PR queue governance evidence while Code
Connect stays disabled.

The Fixed-Item Calibration frame should show the aFIPC calibration evidence
that compares candidate probability tensors on a fixed evaluation-item subset.
It should include the fixed-item count, best candidate, item-fit risk,
`dimension_diagnostics.json` output artifact, and the reproducible
`fast-mlsirm diagnose-fixed-item-calibration` source command.

## aFIPC Product Design Spec

### Information Architecture

Commercial Readiness > Buyer Evidence > Calibration Quality > Fixed-Item
Calibration.

- Inputs: `responses.npy`, repeated candidate probability tensors, and optional
  `fixed_items.npy` as a boolean mask or item-index vector.
- Processing: validate item type, response-process metadata, candidate tensor
  shapes, fixed-item coverage, and kaefa-style item-fit penalty.
- Outputs: `dimension_diagnostics.json`, `best_candidate`,
  `calibration_score`, fixed-item observed counts, and fixed-item outfit
  summaries.

### 화면정의서

- Primary user: evaluation owner or buyer reviewer.
- Primary task: decide which candidate model is acceptable for fixed evaluation
  items.
- Required controls: responses selector, repeated candidate inputs, optional
  fixed item mask/index selector, item type, response process, and item-fit
  penalty weight.
- Empty state: no candidates, invalid fixed-item selection, or no observed
  fixed-item responses must produce explicit validation errors instead of
  blank report content.

### Key Screen

`08-fixed-item-calibration` is the key screen. It must show the fixed-item
subset, selected model, item-fit risk metric, output artifact, source command,
and buyer-facing interpretation. No placeholder-only section is acceptable.

### Wireframe

- Header: method title, aFIPC + kaefa badge, and Code Connect disabled badge.
- Metric row: fixed items, best model, item-fit risk, and output artifact.
- Left panel: calibration inputs and validation boundaries.
- Right panel: buyer decision and interpretation.
- Footer: reproducible CLI command and output path.

### User Stories

- As an evaluation owner, I want fixed-item anchors included in model selection
  so drift in evaluation items is visible.
- As a buyer reviewer, I want the best candidate and item-fit penalty
  summarized in one screen so I can make a go/no-go decision.
- As a release operator, I want explicit validation errors for empty fixed-item
  evidence so a blank report cannot pass procurement review.

## Generated Artifact

The current buyer-review design file is
[fast-mlsirm 20B Buyer Review Screens](https://www.figma.com/design/qD34PfMH8Kr41tFdqLCkem).
This URL is optional evidence, not a CI dependency. If the file is unavailable
in a future environment, regenerate it from
`examples/enterprise_demo/figma_design_packet.json`; Code Connect remains
disabled.

## Handoff

The canonical machine-readable packet is
`examples/enterprise_demo/figma_design_packet.json`. It records the intended
frames, source artifacts, the optional `figma_artifact_url`, and the explicit
`code_connect: false` constraint. `scripts/build_figma_evidence_sync.py` reads
that packet and, when available, an exported live metadata snapshot to create
`figma_evidence_sync_manifest.json` and `figma_evidence_sync_report.html`.
The IRT Stability Review frame should point to
`docs/irt_stability_product_design.md` and `tests/test_irt_stability.py` so
scientific-equation stability is represented as a real product-review surface,
not an empty section.

If a Figma MCP server is connected in a future run, create the file from this
packet using Figma Plugin API calls after loading `figma-use` and
`figma-generate-design`.
