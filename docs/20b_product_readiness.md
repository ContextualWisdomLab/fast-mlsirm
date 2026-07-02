# KRW 2,000,000,000 Product Readiness

## Position

This document defines the extra product evidence required before presenting
`fast-mlsirm` as a KRW 2,000,000,000 enterprise procurement candidate.
It extends `docs/enterprise_sales_readiness.md`; it is not a valuation
guarantee, regulated-use approval, or promise that a buyer will close.

The sellable unit remains the local Python/Rust MLSIRM package plus buyer
evidence. It is not split into a new library or submodule because the current
proof surface is documentation, examples, JSON manifests, and release gates
around the existing package. A separate library becomes appropriate only when a
new independently versioned runtime, hosted service, or customer-facing SDK has
its own release cadence and consumers.

## Buyer-Facing Product Standard

A KRW 2,000,000,000 candidate must show all of the following on the exact
commit and artifact being offered:

- release acceptance evidence from `scripts/release_acceptance.py`;
- enterprise sales evidence from `scripts/sales_readiness.py`;
- Product Design evidence for the buyer workflow and report review journey;
- Figma-ready design packet with Code Connect explicitly disabled;
- Data Analytics evidence for ROI assumptions, KPI definitions, and benchmark
  scenarios;
- a synthetic enterprise demo path that avoids customer response data;
- support, security, scope, formula-contract, and non-goal boundaries.
- a buyer evidence completion scorecard for release acceptance, HTML report
  CSP, CLI stack-trace safety, report table accessibility, and Figma buyer
  review coverage.
- a portable buyer evidence packet containing distribution artifacts,
  acceptance output, sales-readiness output, product manifests, documentation,
  SHA256 digests, and a standalone HTML review report.

## Product Design Scope

The buyer workflow is:

1. install the package from the offered wheel;
2. run synthetic or buyer-provided response data through fitting;
3. inspect fit diagnostics, dimensionality diagnostics, and response-process
   diagnostics;
4. export standalone HTML reports;
5. package acceptance JSON, reports, versions, backend, and runtime evidence for
   procurement review.

The visual product work should document this flow in `docs/buyer_demo_storyboard.md`
and `docs/figma_product_design_packet.md`. Figma Code Connect must stay out of
scope until the product has a stable app component library worth mapping.

## Data Analytics Scope

The analytics package must define:

- ROI driver metrics and caveats;
- benchmark scenarios and runtime evidence expectations;
- acceptance, reproducibility, backend, and report-completeness KPIs;
- go/no-go thresholds that distinguish product evidence from sales claims.

The initial analytics evidence is stored under `examples/enterprise_demo/`.
It uses synthetic examples only. Real buyer data validation remains a customer
implementation activity.

## Go/No-Go

The candidate is a `go` for KRW 2,000,000,000 product-readiness review only
when:

- `scripts/sales_readiness.py --require-20b-product` exits with status `ok`;
- every required product, design, analytics, and demo artifact exists;
- ROI and benchmark manifests are parseable JSON and match the target contract
  value;
- `product_completion_manifest.json` contains all required hardening checks
  with `go` status;
- `scripts/build_buyer_packet.py` can create a buyer evidence packet and
  `scripts/sales_readiness.py --require-buyer-packet` can validate the packet
  manifest when a packet is part of the offer;
- `buyer_evidence_report.html` summarizes coverage, contract value, source
  commit, artifact count, ZIP digest, and artifact digests for human review;
- the Figma design packet declares `code_connect: false`;
- the package acceptance evidence still passes the normal enterprise gate.

## Non-Goals

This product-readiness standard does not add:

- hosted SaaS tenancy, authentication, billing, or audit logs;
- a separate library or submodule;
- a package registry, signing service, or external artifact repository;
- Figma Code Connect;
- regulated clinical, hiring, educational placement, or certification claims;
- new MLSIRM formulas, diagnostics semantics, or estimator scope.

## External Standards Used For The Gate

The gate references current public standards as procurement-language anchors:

- OWASP ASVS 5.0.0 for application security verification terminology;
- W3C WCAG 2.2 for accessibility expectations in buyer-facing reports and
  prototype surfaces;
- SLSA for software supply-chain integrity language;
- OpenSSF Scorecard for open source security posture language.

Those standards guide evidence wording. They are not claimed as full
certification unless a separate audit produces that evidence.
