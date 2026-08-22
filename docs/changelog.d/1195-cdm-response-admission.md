# Cognitive-diagnosis response admission hardening

## Fixed

- Reject complex or non-real-numeric response storage before real-valued marshalling across DINA/DINO, G-DINA, PVAF Q-matrix validation, Wald item-model selection, higher-order DINA/G-DINA, and shared/per-step-Q sequential G-DINA entry points so observed evidence cannot be silently projected onto different data.
- Require accepted numeric response evidence to round-trip exactly through the `float64` Rust boundary, rejecting extended-precision or integer values whose identity would change during marshalling while preserving exact values and `NaN` missingness.
- Normalize model and stopping controls before caller response materialization across the CDM calibration, validation, model-selection, higher-order, and sequential entry points, preserving a consistent fail-closed control boundary.
- Preserve binary and ordered-category `NaN` missingness, Q-matrix validation, and Rust ownership of CDM likelihoods, marginal-ML EM, parameter estimation, classification, model-selection/validation statistics, higher-order structure, sequential-category arithmetic, and convergence.
