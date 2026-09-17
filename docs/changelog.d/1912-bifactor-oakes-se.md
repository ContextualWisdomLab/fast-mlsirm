# Bifactor GRM observed-information standard errors (Oakes)

## Added

Stage-3 standard errors for the single-group polytomous bifactor graded
response model (stage 3 of #1912): `mlsirm_core::bifactor_oakes::
bifactor_oakes_se` (with `BifactorOakesConfig`) returns the full
item-parameter observed information, its inverse vcov, and standard errors
via the Oakes (1999, eq. 6) identity, evaluated at given item parameters.
The complete-data gradient and Hessian are analytic
(`poly::grm_node_hessian`, cross-checked against central finite differences
in tests); the cross term re-runs the Gibbons-Hedeker reduced E-step once
per free parameter. The assembly is written behind a `PosteriorProvider`
trait so the stage-2 multigroup calibration reuses it with group-specific
E-steps. A non-positive-definite information matrix is reported with
`positive_definite = false` and a `non_pd_reason`, while `vcov`/`se` are
`None` — never a generalized inverse or any other substitute (#1912
acceptance criterion 3). Python surface:
`fast_mlsirm.bifactor_grm.bifactor_oakes_se` returning `BifactorOakesSe`
(quadrature counts and the cross-term step are required caller arguments).

Evidence: analytic-vs-finite-difference Q-Hessian agreement; Oakes
information vs numerical Hessian of the exact marginal log-likelihood on a
tiny problem; mean-SE vs empirical-SD agreement over 100 simulation
replicates at recovery scale (worst 0.235, Monte Carlo noise ~7%);
small-grid convergence stabilization; mirt `SE.type = "Oakes"`
agreement on the committed fixture (SEs 8.4e-3, vcov 1.9e-2 at matched
quadpts = 15); study-settings grid convergence at 121 vs. 241 Gauss-Hermite
nodes per dimension (#1929's node-count cap removal made this grid
reachable) — Oakes SEs agree to `maxRel|dSE| = 8.76e-9`, run once locally
in `bifactor_oakes_calibration::study_settings_se_converges_at_121_vs_241_nodes`
(`#[ignore]`d as long-running, ~46 min at `q=241`).
