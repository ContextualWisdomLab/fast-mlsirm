# Canonical product and architecture documentation baseline

## Changed

- Replaced the stale MVP-only PRD/TRD authority with canonical `docs/PRD.md` and `docs/TRD.md` requirements covering the current measurement, scoring, rubric/item-generation, model-selection, scientific-evidence, interoperability, security, lifecycle, and release boundaries.
- Added root `ARCHITECTURE.md`, a status-bearing ADR corpus, reviewable PlantUML component/sequence/state/deployment views, a logical reusable-domain ERD, and requirements/research traceability matrices.
- Added a machine-checkable documentation contract that fails when canonical architecture files disappear, ADR statuses are invalid, root architecture links break, or the hosted-product ownership boundary drifts.
- Explicitly deprecated the original narrow `docs/prd_trd_summary.md` as an authoritative requirements source while retaining its historical MLS2PLM MVP context.
