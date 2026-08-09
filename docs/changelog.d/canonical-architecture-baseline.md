# Canonical product and architecture documentation baseline

## Changed

- Replaced the stale MVP-only PRD/TRD authority with canonical `docs/PRD.md` and `docs/TRD.md` requirements covering the current measurement, scoring, rubric/item-generation, model-selection, scientific-evidence, interoperability, security, lifecycle, and release boundaries.
- Added root `ARCHITECTURE.md`, a status-bearing ADR corpus, reviewable PlantUML component/sequence/state/deployment views, a logical reusable-domain ERD, and requirements/research traceability matrices.
- Added a canonical documentation authority index, explicit implementation-maturity/completeness matrix, and machine-checkable documentation contract so missing or stale PRD/TRD/ADR/UML/ERD/traceability/security artifacts remain visible release-maintenance debt rather than silently drifting.
- Added a reusable-core threat model covering provider/JSON replay, native/PyO3 input boundaries, resource and non-finite numerical failures, GPU evidence spoofing, supply-chain/self-modifying CI, credential separation, benchmark contamination, privacy/purpose limitation, and scientific-interpretation abuse while leaving hosted HTTP/session/tenant/database threats downstream.
- Added durable ADRs for converging future Rust-backed features on one canonical PyO3/public-export registry and for preserving legitimate sensitive-data linkage through purpose limitation and minimization rather than blanket masking that changes the measurement design.
- Extended requirements traceability with the conversation-wide invariants that human/LLM judges are fallible raters, correlation is not parameter recovery/absolute agreement, latent-space interaction follows substantive dimension/testlet/facet diagnosis, reference-free is not truth-free, and psychometric discrimination is not business or safety criticality.
- Explicitly deprecated the original narrow `docs/prd_trd_summary.md` as an authoritative requirements source while retaining its historical MLS2PLM MVP context.
