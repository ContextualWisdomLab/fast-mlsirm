### Added

- Rust-native continuous-indicator bifactor scoreability diagnostics: ECV-SS,
  ECV-SG, ECV-GS, item ECV, strict-pattern PUC, omega total, omega
  hierarchical, and construct replicability H.
- An explicitly named logistic latent-response conversion for fitted orthogonal
  bifactor slopes. Its omega values are documented as continuous
  latent-response coefficients, not categorical observed-score reliability.
- A modular PyO3 `_bifactor_core` surface and immutable typed Python API:
  `bifactor_scoreability`, `bifactor_scoreability_from_logit_slopes`, and
  `BifactorScoreabilityResult`. Python validates shapes and marshals results;
  all scoreability arithmetic remains in Rust.
- Fail-closed structural validation requiring every item to load on the
  declared general factor, uniquenesses in `[0, 1]`, and the standardized
  identity `sum(lambda^2) + uniqueness = 1` within `1e-8`.
- Formula-oracle, Rust/Python parity, structural, numerical-stability,
  logistic-conversion, and package-export tests plus buyer-facing
  interpretation boundaries.
