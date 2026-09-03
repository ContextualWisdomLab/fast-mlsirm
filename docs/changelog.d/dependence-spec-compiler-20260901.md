# Dependence specification compiler

## Added

- Added a typed, non-numerical Model Specification bounded context that composes response-kernel, dimensional, generalized-mixed, identification, recovery, and dependence metadata and automatically materializes LSIRM, MLSIRM, and DLSJM candidates as `supported`, `research_candidate`, or `unsupported` without silently falling back to the base model. DLSJM keeps distinct item- and person-dependence spaces, and temporal evolution remains TEPP-owned.
- Replaced free-form generalized-mixed membership labels with explicit classification, multiplicity, weight-authority, and classification-axis contracts. Cross-classification remains distinct from multiple membership; every declared cross-classification axis has a unique identity; model-estimated membership weights require their own named recovery metric before support promotion.
- Versioned the compiled candidate manifest as `fast_mlsirm.model_specification.candidate_manifest@1.0.0` and added a deterministic SHA-256 self-digest over the canonical manifest payload for downstream ACL/provenance checks.
