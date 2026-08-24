# Strengthen polytomous recovery calibration evidence

## Changed

- Extend deterministic GRM, GPCM, and fixed-item-parameter calibration recovery studies to require signed-bias, MAE, finite positive EAP posterior uncertainty, and empirical 95% posterior-interval coverage alongside RMSE, while retaining correlation only as supplementary recovery evidence.
- Keep all production likelihood, marginal-ML/EM, EAP scoring, and uncertainty arithmetic Rust-owned; the added Python calculations are explicit true-parameter recovery-test summaries only.
