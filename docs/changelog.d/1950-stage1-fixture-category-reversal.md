# Stage-1 bifactor GRM mirt fixture regenerated with corrected category order (#1950)

## Fixed

- `tests/fixtures/bifactor_grm_stage1/generate_mirt_fixture.R` compared the
  simulated uniform draw against each graded-response boundary probability
  with `u > p_k`, which reversed the intended category order (higher latent
  trait produced *lower* observed categories). The nested-event identity
  `P(Y >= k) = P(u < p_k)` requires `u < p_k`; regenerated `dataset.csv` and
  `mirt_fixture.json` with the corrected rule and added a category-order
  guard (`cor(theta_g, rowSums(resp)) > 0.3`) so a reversed rule fails fast
  next time instead of only being caught by inspection.
- `crates/mlsirm-core/tests/bifactor_grm_mirt_agreement.rs` still passes
  unchanged: the Rust<->mirt comparison canonicalizes reflection per
  dimension before comparing slopes/intercepts/log-likelihood, so it was
  insensitive to the category-order bug and remains a valid agreement check
  on the corrected fixture.
