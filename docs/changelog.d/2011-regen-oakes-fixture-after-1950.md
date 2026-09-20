# Regenerate stage-3 mirt Oakes fixture after #1950 dataset correction

## Fixed

- `tests/fixtures/bifactor_grm_stage3_oakes/mirt_oakes_fixture.json` still
  carried mirt MLE / Oakes SEs fitted on the pre-#1950 category-reversed
  `dataset.csv`. After #1950 regenerated the stage-1 CSV, the Rust Oakes
  test evaluated information at a foreign MLE and failed PD (cholesky
  pivot 29). Regenerated the fixture with mirt 1.46.1 against the corrected
  dataset; `bifactor_oakes_mirt` passes again (worst SE gap ≈ 9.8e-3).
- Documented the regenerate-when-CSV-changes contract in
  `generate_mirt_oakes_fixture.R`, and fixed the minimal JSON writer so
  character vectors are quoted.
