# Cognitive-diagnosis response admission hardening

## Fixed

- Reject complex or non-real-numeric response storage before real-valued marshalling across DINA/DINO, G-DINA, PVAF Q-matrix validation, Wald item-model selection, higher-order DINA/G-DINA, and shared/per-step-Q sequential G-DINA entry points so observed evidence cannot be silently projected onto different data.
- Normalize model and stopping controls before caller response materialization where those semantic controls already precede the observation boundary, preserving the existing fail-closed control contract.
- Preserve binary and ordered-category `NaN` missingness, Q-matrix validation, and Rust ownership of CDM likelihoods, marginal-ML EM, parameter estimation, classification, model-selection/validation statistics, higher-order structure, sequential-category arithmetic, and convergence.
