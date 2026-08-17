# Governed RAG facets calibration

## Added

- Added the governed RAG facets calibration adapter.
- The adapter reuses the existing MFRM rater-severity/threshold calibration
  machinery (Linacre, 1989; Eckes, 2015; Bock & Aitkin, 1981; Andrich, 1978)
  for RAG evaluation executions. It does not introduce a new psychometric
  estimator or rely on legacy evaluation package implementations (e.g.
  RAGAS-style tooling) as the source of psychometric validity; all
  likelihood/threshold arithmetic is delegated to the existing Rust-backed MFRM
  fit grounded in the primary many-facet Rasch measurement literature. Full
  citations are in `docs/scoring_facets_calibration_handoff.md`.
