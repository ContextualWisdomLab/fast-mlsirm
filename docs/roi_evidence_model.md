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

## Required Evidence

The evidence model is represented by `examples/enterprise_demo/roi_manifest.json`.
The manifest must include:

- `contract_value_krw`;
- `position`;
- non-empty `drivers`;
- non-empty `required_evidence`;
- non-empty `go_no_go`.

## Caveats

Real ROI must be recalculated with buyer-specific hourly rates, workflow
volume, privacy constraints, and validation burden. Synthetic demo evidence
shows reproducibility and product packaging, not customer-specific economic
outcomes.
