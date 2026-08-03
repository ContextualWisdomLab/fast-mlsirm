# Pilot-observation handoff to binary bifactor calibration

## Added

- A factory-sealed, content-addressed `BifactorPilotDesign` assembled by
  `build_bifactor_pilot_design` from the existing replay-verified binary pilot
  design. The artifact records one descriptive general-factor identity over
  every item, preserves each item's governed `query_testlet_id` as its sole
  specific-factor assignment, retains exact missingness and rater provenance,
  and emits copied `responses`, `factor_id`, and `FitConfig` arguments accepted
  directly by the Rust-backed `fast_mlsirm.fit` API.
- The handoff pins `model="BIFAC2PLM"`, `estimator="mmle"`, and
  `latent_dim=1`; caller-tuned numerical settings are accepted only when those
  structural constraints remain intact. It reuses the binary MIRT assembler's
  fail-closed provenance, duplicate-cell, category, observed-support, and dense
  allocation contracts rather than introducing a weaker parallel parser.
- Buyer documentation states the classical all-items general-factor plus
  at-most-one-specific-factor pattern and the downstream identification,
  model-comparison, scoreability, DIF/fairness, recovery, and validity gates.
  A successful handoff is not a fit, scoreability, fairness, or deployment
  claim. Testlet, DIF, and G-theory handoffs remain follow-up slices of issue
  #407.
