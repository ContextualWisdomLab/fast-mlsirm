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
python scripts/build_commercial_release.py \
  --out commercial-release \
  --require-rust \
  --check-import
```

The commercial release builder writes `commercial_release_manifest.json` and
`commercial_release_report.html`, then leaves the acceptance, benchmark, buyer
packet, release index, final gate, and procurement due-diligence artifacts
under the same output directory.

The lower-level checks can also be run manually:

```bash
python scripts/build_benchmark_report.py \
  --acceptance release-acceptance/acceptance_summary.json \
  --out release-acceptance/benchmark

python scripts/sales_readiness.py \
  --acceptance release-acceptance/acceptance_summary.json \
  --dist dist \
  --require-rust \
  --require-20b-product \
  --benchmark-report release-acceptance/benchmark/benchmark_report.json \
  --require-benchmark-report \
  --check-import
```

A portable buyer review packet can be generated after acceptance and sales
readiness output exists:

```bash
python scripts/build_buyer_packet.py \
  --acceptance release-acceptance/acceptance_summary.json \
  --sales-readiness release-acceptance/sales_readiness_manifest.json \
  --dist dist \
  --benchmark-report release-acceptance/benchmark/benchmark_report.json \
  --out buyer-evidence-packet

python scripts/build_release_evidence_index.py \
  --acceptance release-acceptance/acceptance_summary.json \
  --sales-readiness release-acceptance/sales_readiness_manifest.json \
  --dist dist \
  --benchmark-report release-acceptance/benchmark/benchmark_report.json \
  --buyer-packet-manifest buyer-evidence-packet/buyer_evidence_manifest.json \
  --out release-evidence-index

python scripts/build_procurement_due_diligence.py \
  --dist dist \
  --commercial-release-manifest commercial-release/commercial_release_manifest.json \
  --out procurement-due-diligence
```

The command writes `buyer_evidence_manifest.json`,
`buyer_evidence_report.html`, and `fast_mlsirm_buyer_evidence_packet.zip`.
The release evidence command writes `release_evidence_index.json` and
`release_evidence_index.html` for dist hash, acceptance, benchmark,
sales-readiness, and buyer-packet review. The procurement command writes
`procurement_due_diligence_manifest.json` and
`procurement_due_diligence_report.html` for package metadata, policy-file,
commercial-release, GitHub snapshot, and SHA256 report review.
