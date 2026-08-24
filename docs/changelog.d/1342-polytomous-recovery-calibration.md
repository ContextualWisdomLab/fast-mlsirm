# Strengthen polytomous recovery calibration evidence

## Changed

- Extend deterministic GRM, GPCM, CAT, and fixed-item-parameter recovery studies to require signed bias, MAE, finite positive Rust-returned posterior uncertainty, and empirical coverage of the normal-approximation interval `theta_eap ± 1.96 * theta_sd` alongside RMSE, while retaining correlation only as supplementary recovery evidence.
- Preserve CAT's independent adaptive-efficiency gate on mean administered items, so uncertainty calibration and error recovery cannot mask a fallback to non-adaptive item selection.
- Keep all production likelihood, marginal-ML/EM, EAP/CAT scoring, item-information/selection, stopping, and uncertainty arithmetic Rust-owned; the added Python calculations are explicit true-parameter recovery-test summaries only.
