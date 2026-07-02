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
2. Synthetic Demo Run
3. Fit Diagnostics
4. Dimensionality Review
5. Report Export
6. Procurement Packet

Each frame should include:

- a concise buyer task title;
- the artifact being inspected;
- the go/no-go signal;
- the source file or command that produces the artifact.

## Handoff

The canonical machine-readable packet is
`examples/enterprise_demo/figma_design_packet.json`. It records the intended
frames, source artifacts, and the explicit `code_connect: false` constraint.

If a Figma MCP server is connected in a future run, create the file from this
packet using Figma Plugin API calls after loading `figma-use` and
`figma-generate-design`.
