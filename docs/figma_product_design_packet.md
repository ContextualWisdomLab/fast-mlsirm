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
single-command stage summary and human review surface.

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
`code_connect: false` constraint.

If a Figma MCP server is connected in a future run, create the file from this
packet using Figma Plugin API calls after loading `figma-use` and
`figma-generate-design`.
