# Enterprise Demo Evidence

This directory contains synthetic, non-customer evidence for the
KRW 2,000,000,000 product-readiness gate.

Files:

- `roi_manifest.json`: Data Analytics ROI driver model and caveats.
- `benchmark_manifest.json`: benchmark scenario contract for acceptance runs.
- `figma_design_packet.json`: Product Design and Figma screen packet with
  Code Connect disabled.
- `product_completion_manifest.json`: compact go/no-go scorecard for the
  current hardening evidence.

These files are checked by:

```bash
python scripts/sales_readiness.py \
  --acceptance release-acceptance/acceptance_summary.json \
  --dist dist \
  --require-rust \
  --require-20b-product \
  --check-import
```
