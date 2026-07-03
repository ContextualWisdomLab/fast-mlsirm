# 20B Procurement Due-Diligence Design

## Goal

Add buyer-review evidence for procurement and supply-chain due diligence
without changing MLSIRM formulas, diagnostics semantics, estimator scope, or
package boundaries.

## Decision

Keep the work inside the existing repository as
`scripts/build_procurement_due_diligence.py`. A separate library, submodule, or
hosted dashboard is not justified because this feature is an evidence builder
over the current package, distribution artifacts, policy files, GitHub state,
and commercial release manifest. Figma Code Connect remains out of scope.

The existing Figma buyer-review design remains the product design artifact. The
Procurement Packet frame should now point to
`procurement_due_diligence_manifest.json` and
`procurement_due_diligence_report.html` in addition to the buyer packet, release
evidence index, and commercial release report.

## Evidence Contract

The due-diligence builder must check:

- wheel and source distribution artifacts exist;
- wheel metadata includes `METADATA`, `WHEEL`, and `RECORD`;
- source distribution metadata includes `PKG-INFO`;
- pyproject, wheel, and source distribution versions match;
- required policy and readiness documents exist;
- `.github/workflows/ci.yml` exists;
- commercial release manifest status, contract value, wheel, source
  distribution, and final sales-readiness evidence are present;
- GitHub repository, open PR, and release snapshot state is recorded, or
  explicitly marked offline for deterministic local verification.

It must output:

- `procurement_due_diligence_manifest.json` for machine review;
- `procurement_due_diligence_report.html` for human procurement review.

The HTML report must include a restrictive CSP meta tag, decision summary, and
a focusable due-diligence check table.

## Analytics Scope

This evidence supports reproducibility, package metadata, policy coverage,
repository-state review, and digest verification. It is not a valuation
guarantee, security certification, regulated-use approval, or package signing
attestation.

## Go/No-Go

The due-diligence run is `ok` only when all package, policy,
commercial-release, and GitHub snapshot checks pass and the report SHA256 can
be validated by `scripts/sales_readiness.py
--require-procurement-due-diligence`.

## Non-Goals

- New MLSIRM formulas, diagnostics semantics, or estimator scope.
- A hosted product surface, dashboard, login, billing, or customer-data upload.
- Artifact signing, package registry publication, SBOM generation, or external
  attestation service.
- Figma Code Connect.
