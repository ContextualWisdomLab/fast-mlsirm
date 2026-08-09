# fast-mlsirm PRD/TRD Summary

Status: **Compatibility pointer**  
Last reconciled: **2026-08-09**

The former content of this file described an early MLS2PLM MVP in which NumPy was the default implementation and ordinal scoring, adaptive testing, rubric generation, automated scoring, and broader measurement workflows were explicitly out of scope. That description is materially stale relative to protected `main` and must not be used as the product or technical specification.

The authoritative documents are now:

- [Product Requirements Document](product_requirements.md)
- [Technical Requirements Document](technical_requirements.md)
- [Root architecture](../ARCHITECTURE.md)
- [Architecture Decision Records](adr/README.md)
- [Architecture/UML/logical ERD diagrams](architecture/diagrams.md)
- [Requirements and evidence traceability](traceability_matrix.md)

## Current concise boundary

`fast-mlsirm` is a reusable, domain-neutral measurement and psychometric computation layer. It provides versioned assessment/rubric/scoring contracts, governed rubric/item-generation boundaries, human/automated scoring evidence, Rust-first calibration/diagnostics, model comparison and scoreability evidence, recovery/simulation, linking, DIF/invariance/fairness-related utilities, and deterministic reports within the capabilities exposed by the current package.

Rust is the production numerical authority. Python owns public contracts, validation, orchestration, retained reference/fallback paths where explicitly supported, and reporting. PyO3 is the reviewed bridge. GPU execution is a parity-gated device path rather than a separate scientific model.

The package is not the hosted assessment product and does not own participant/session/consent persistence or hosted-product authorization/deployment. Those responsibilities belong downstream to `ContextualWisdomLab/psychometrics-commons`.

## Current major product directions

The authoritative PRD/TRD and traceability matrix distinguish released work from active/planned work. In particular:

- a generated item is untrusted until structural/provenance/source checks and later semantic/psychometric screening succeed;
- humans and LLM judges are fallible raters rather than truth by identity;
- correlated multidimensional, bifactor, higher-order, testlet/two-tier, multifaceted, and latent-space structures answer different scientific questions and require relation-safe comparison;
- parameter recovery uses bias/MAE/RMSE/coverage and appropriate scale/alignment, not correlation alone;
- multilevel, multiple-membership, cross-classified, and temporal structure must be preserved when scientifically relevant; and
- release claims require exact-head CI/security/coverage/package/scientific/provenance/review evidence.

This pointer is intentionally short. New normative requirements belong in the PRD/TRD or an ADR, not in another standalone summary that can silently diverge.
