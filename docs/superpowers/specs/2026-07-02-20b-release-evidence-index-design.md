# 20B Release Evidence Index Design

## Goal

Create a release-level evidence index for KRW 2,000,000,000 procurement review.
The index should let a buyer, reviewer, or seller confirm that the offered
wheel/source distribution, acceptance run, benchmark report, sales-readiness
manifest, and buyer packet all belong to the same source commit and have
verifiable SHA256 digests.

## Product Decision

Do not split a new library or submodule for this work. The current product sold
unit is the existing Python/Rust package plus release evidence. A separate
library would add versioning and audit complexity without improving buyer
review. Splitting becomes appropriate only when a separately consumed runtime,
hosted service, or SDK needs an independent release cadence.

Figma Code Connect remains out of scope. The existing Figma buyer-review file is
used as product-design evidence for the procurement flow, not as a source of
generated application code.

## Evidence Contract

The release index must consume:

- `dist/*.whl` and `dist/*.tar.gz`;
- `acceptance_summary.json`;
- `benchmark_report.json` plus its HTML report;
- `sales_readiness_manifest.json`;
- `buyer_evidence_manifest.json` plus its ZIP and HTML report.

It must output:

- `release_evidence_index.json` for machine validation;
- `release_evidence_index.html` for human procurement review.

The JSON must include source commit, package version, generated timestamp,
target contract value, coverage booleans, dist artifact digests, acceptance
status, benchmark budget status, sales-readiness status, buyer packet digest,
HTML report digest, and failures.

The HTML must include a restrictive CSP meta tag, focusable review tables, and
digest rows that can be inspected without reading raw JSON.

## Go/No-Go

The release index is a `go` only when:

- wheel and source distribution artifacts exist and have SHA256 digests;
- acceptance, benchmark, sales-readiness, buyer packet, and release index
  coverage are present;
- acceptance, benchmark, sales-readiness, and buyer packet statuses are `ok`;
- benchmark HTML, buyer packet ZIP, buyer packet HTML, and release index HTML
  digests match their manifests;
- `scripts/sales_readiness.py --require-release-evidence-index` validates the
  generated index.

## Non-Goals

- Artifact signing, package registry publication, or SLSA attestation service.
- Hosted dashboard or customer data upload flow.
- Model formula, estimator, or diagnostics semantic changes.
- Recursive packet/index hashing where a packet includes an index that validates
  that same packet.
