# KRW 2,000,000,000 Product Readiness Design

## Goal

Make `fast-mlsirm` presentable as a KRW 2,000,000,000 enterprise procurement
candidate by adding buyer-facing product, design, analytics, and automated
evidence gates around the existing Python/Rust package.

## Constraints

- Do not change MLSIRM formulas, fit diagnostics semantics, estimators, or
  simulation contracts.
- Do not use Figma Code Connect.
- Do not add a hosted SaaS, authentication, billing, tenancy, or audit-log
  layer.
- Do not split a new library or submodule unless an independently versioned
  runtime or SDK exists. It does not exist in this scope.
- Use synthetic demo evidence only.
- Treat review delay as operational friction, not a blocker.

## Approach

Use the existing release and enterprise sales readiness machinery as the
foundation. Add a stricter `--require-20b-product` mode to
`scripts/sales_readiness.py` that checks the extra product package: product
readiness docs, buyer demo storyboard, Figma design packet, ROI model, benchmark
manifest, and synthetic demo artifacts.

## Product Design

The buyer journey is a static review flow:

1. package evidence;
2. synthetic demo run;
3. fit diagnostics;
4. dimensionality review;
5. report export;
6. procurement packet.

This is enough for the current local package. A hosted interactive dashboard is
deferred until there is a buyer-backed reason.

## Data Analytics

The data work defines ROI and benchmark evidence, not revenue claims. The
required evidence is a parseable ROI manifest and benchmark manifest with
explicit caveats and go/no-go fields.

## Figma

The Figma output is represented by a machine-readable design packet and
storyboard. If Figma MCP is connected later, the packet can become a static
board. Code Connect stays disabled.

## Testing

Tests must cover:

- the new `--require-20b-product` parser flag;
- successful validation against repository artifacts;
- failure when a required 20B product artifact is missing;
- failure when the Figma design packet enables Code Connect.

## Acceptance

The work is complete when:

- local tests pass;
- Rust tests still pass;
- package build and twine check pass;
- release acceptance and sales readiness pass with `--require-20b-product`;
- CI runs the new product readiness gate;
- PR is created, review feedback is handled, and the checked head is merged.
