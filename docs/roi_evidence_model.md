# ROI Evidence Model

## Purpose

This model turns the KRW 2,000,000,000 product-readiness claim into auditable
evidence. It does not assert valuation. It defines the evidence a buyer would
need to decide whether the package can justify a high-value procurement review.

## Driver Metrics

- **Analyst hours saved:** time avoided by repeatable simulation, fitting,
  diagnostics, and report generation.
- **Engineering hours saved:** time avoided by packaged Python/Rust install,
  CLI automation, and acceptance manifests.
- **Reproducibility:** percentage of buyer-review evidence produced from a
  scripted synthetic path.
- **Runtime confidence:** acceptance runtime and benchmark scenario status.
- **Governance confidence:** support, security, formula-scope, and non-goal
  documentation present in the release artifact.
- **Supply-chain confidence:** wheel/source metadata, policy files,
  commercial-release integrity, GitHub snapshot state, and report digest
  evidence present for procurement review.
- **Queue governance confidence:** open PR review state, stale or
  changes-requested work, release-scope conflicts, and report digest evidence
  present for procurement review.
- **Design evidence confidence:** Code Connect-disabled Figma packet, required
  buyer-evidence tokens, optional live metadata snapshot status, and report
  digest evidence present for procurement review.

## Required Evidence

The evidence model is represented by `examples/enterprise_demo/roi_manifest.json`.
The manifest must include:

- `contract_value_krw`;
- `position`;
- non-empty `drivers`;
- non-empty `required_evidence`;
- non-empty `go_no_go`.

The product completion scorecard is represented by
`examples/enterprise_demo/product_completion_manifest.json`. It must include
`go` evidence for release acceptance, HTML report CSP, CLI stack-trace safety,
report table accessibility, the Figma buyer-review artifact, the buyer
evidence packet, the buyer evidence HTML review, the automated benchmark
report, and the release evidence index.
The same scorecard also requires the commercial release builder so a buyer can
produce the full evidence set from one command, plus procurement due diligence
so package metadata, policy files, GitHub state, commercial release evidence,
and report SHA256 can be reviewed independently. It also requires PR queue
governance so open review work, stale PRs, and release-scope conflicts are
classified instead of hidden.
It also requires Figma evidence sync so the design packet is checked against
the same buyer packet, release evidence index, procurement due-diligence, and
PR queue governance evidence before the Figma artifact is used in a buyer
review.

The benchmark evidence is represented by `benchmark_report.json` generated from
`scripts/build_benchmark_report.py`. The companion `benchmark_report.html`
presents runtime-budget status, total duration, command timings, backend
coverage, required artifact coverage, and caveats.

The buyer evidence packet is represented by `buyer_evidence_manifest.json`
generated from `scripts/build_buyer_packet.py`. It records artifact coverage,
SHA256 digests, `source_commit`, `generated_at`, and the target
`contract_value_krw`. The companion `buyer_evidence_report.html` presents the
same coverage and digest evidence as KPI cards and review tables.

The release evidence index is represented by `release_evidence_index.json`
generated from `scripts/build_release_evidence_index.py`. The companion
`release_evidence_index.html` presents distribution artifact hashes, acceptance
status, benchmark status, sales-readiness status, buyer packet digest, source
commit, and required evidence coverage for procurement review.

The top-level buyer review is represented by `commercial_release_manifest.json`
generated from `scripts/build_commercial_release.py`. The companion
`commercial_release_report.html` presents stage status, command duration,
failed-stage detail, artifact paths, contract value, source commit, and SHA256
evidence from dist build, acceptance, benchmark, sales-readiness, buyer packet,
release index, and final gate stages.

The procurement due-diligence evidence is represented by
`procurement_due_diligence_manifest.json` generated from
`scripts/build_procurement_due_diligence.py`. The companion
`procurement_due_diligence_report.html` presents package metadata, policy-file
checks, commercial-release integrity, GitHub snapshot state, failed checks, and
report SHA256 evidence for procurement review.

The PR queue governance evidence is represented by
`pr_queue_governance_manifest.json` generated from
`scripts/build_pr_queue_governance.py`. The companion
`pr_queue_governance_report.html` presents open PR count, reviewDecision,
mergeStateStatus, stale and changes-requested counts, release-scope conflict
classification, and report SHA256 evidence for procurement review.

The Figma evidence sync is represented by `figma_evidence_sync_manifest.json`
generated from `scripts/build_figma_evidence_sync.py`. The companion
`figma_evidence_sync_report.html` presents Code Connect-disabled status,
required frame coverage, required buyer-evidence token coverage, optional
metadata snapshot status, and report SHA256 evidence for procurement review.

## Caveats

Real ROI must be recalculated with buyer-specific hourly rates, workflow
volume, privacy constraints, and validation burden. Synthetic demo evidence
shows reproducibility and product packaging, not customer-specific economic
outcomes.
