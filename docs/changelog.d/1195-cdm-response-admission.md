# Cognitive-diagnosis response admission hardening

## Fixed

- Reject complex or non-real-numeric DINA/DINO and G-DINA response storage before real-valued marshalling so observed 0/1 evidence cannot be silently projected onto different data.
- Normalize model and stopping controls before caller response materialization so rejected controls cannot trigger caller array protocols first.
- Preserve 0/1/`NaN` response semantics, Q-matrix validation, and Rust ownership of DINA/DINO/G-DINA likelihood, marginal-ML EM, parameter estimation, classification, and convergence arithmetic.
