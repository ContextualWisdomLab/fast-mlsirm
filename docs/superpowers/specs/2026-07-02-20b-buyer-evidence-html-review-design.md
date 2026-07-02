# KRW 2,000,000,000 Buyer Evidence HTML Review Design

## Objective

Make the buyer evidence packet directly reviewable by non-engineering
procurement stakeholders. The packet builder must generate a standalone HTML
review report next to the existing JSON manifest and zip so buyers can inspect
coverage, contract value, source commit, artifact count, zip digest, and file
digests without reading raw JSON first.

## Boundaries

- Do not add a separate library, submodule, dashboard, frontend app, hosted
  service, or package split.
- Do not use Figma Code Connect.
- Do not change formulas, estimators, diagnostics, or model interpretation.
- Use only Python standard library modules already appropriate for the packet
  builder path.
- Keep buyer-packet validation optional unless the caller passes the packet
  validation flags.

## Product Design

The existing Figma file
`https://www.figma.com/design/qD34PfMH8Kr41tFdqLCkem` remains the visual
reference. The `06-procurement-packet` frame maps to three repo-generated
outputs:

- `buyer_evidence_manifest.json`;
- `fast_mlsirm_buyer_evidence_packet.zip`;
- `buyer_evidence_report.html`.

The HTML report is the human-readable review surface for the same evidence the
manifest records. It must be accessible by keyboard and readable as a static
file with no JavaScript.

## Data Analytics

The report presents procurement evidence as KPI cards and digest tables:

- target `contract_value_krw`;
- `artifact_count`;
- `source_commit`;
- packet `zip_sha256`;
- required coverage status;
- per-file byte size and SHA256 evidence.

This is a readiness and provenance report, not a valuation guarantee,
customer-specific ROI calculation, or regulatory certification.

## Ponytail / Architecture Decision

Do not split a new library or submodule for this wave. The feature is a narrow
artifact-generation extension of `scripts/build_buyer_packet.py`, and a split
would increase review and packaging surface without improving buyer evidence.
Keep the implementation in the existing script and validate it through
`scripts/sales_readiness.py`.

## Acceptance

- Packet builder writes `buyer_evidence_report.html`.
- HTML includes a CSP meta tag and keyboard-focusable evidence tables.
- Manifest records `report_file` and `report_sha256`.
- Zip includes the HTML report.
- Sales readiness can validate HTML report presence and SHA256 when
  `--require-buyer-packet` is supplied.
- Documentation links the Figma procurement-packet frame to the generated HTML
  review surface.
