# fast-mlsirm PRD/TRD Summary

This file is an index. The former combined PRD/TRD text described an early NumPy-first MLS2PLM MVP and had become materially stale: the current repository is Rust-first for production psychometric arithmetic and now includes broader scoring, rubric/item-generation, enterprise measurement, accessibility, release-evidence and integration contracts.

Authoritative current documents:

- Product requirements: [`docs/PRD.md`](PRD.md)
- Technical requirements: [`docs/TRD.md`](TRD.md)
- Architecture: [`../ARCHITECTURE.md`](../ARCHITECTURE.md)
- UML: [`docs/UML.md`](UML.md)
- Persistence-neutral logical ERD: [`docs/ERD.md`](ERD.md)
- Architecture/scientific-governance ADR: [`docs/adr/ADR-0001-product-boundaries-and-scientific-governance.md`](adr/ADR-0001-product-boundaries-and-scientific-governance.md)
- Documentation coverage and residual gaps: [`docs/documentation_coverage_matrix.md`](documentation_coverage_matrix.md)

## Product boundary in one paragraph

`fast-mlsirm` is the reusable, domain-neutral psychometric measurement and Rust-first numerical-computation layer. It owns versioned assessment/rubric/scoring/evidence contracts, simulation/estimation/diagnostics/recovery/model selection, linking/DIF/fairness, scoring calibration, and reusable measurement workflows. The canonical hosted application is `ContextualWisdomLab/psychometrics-commons`; hosted participant/session/consent lifecycle, tenant persistence/ORMs, identity, HTTP/admin APIs, UI and deployment composition are downstream responsibilities.

## Current engineering principles

1. Production psychometric arithmetic is Rust-first; Python validates, marshals, orchestrates and reports.
2. Model selection is relation-safe and evidence-based; no preference is claimed without appropriate distinguishability/predictive evidence.
3. True-parameter recovery uses aligned bias/MAE/RMSE/coverage/convergence rather than correlation-only evidence.
4. Multilevel, cross-classified, multiple-membership and longitudinal/temporal structure are explicit where relevant.
5. Human and AI judges are fallible raters; neither is silently treated as ground truth.
6. Rubric/item/scoring artifacts are versioned and provenance-bound; operational versions are immutable.
7. PII protection uses purpose-bound authorization, isolation, selective disclosure and encryption rather than blanket masking that destroys valid measurement workflows.
8. Release claims require one exact integrated head with CI, security, coverage, package, provenance, recovery/validation, review, rollback and changelog evidence.
