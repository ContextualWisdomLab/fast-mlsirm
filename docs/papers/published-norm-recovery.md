# Published NORM parameter-recovery acceptance

## Source and scope

Robitzsch (2024), Equation 2, Section 3.1 and Appendix Table A1 publish a
10-item NORM data-generating model with standard-normal ability and sample
sizes including N=2000. The table coefficients originated in empirical fits,
but the author explicitly used these fixed coefficients as simulation truth;
this benchmark uses that published generating bank, not an empirical-fit
accuracy oracle. The full body, table and equation were inspected at the
publisher.[1]

The test preserves all ten printed NORM rows (two decimal precision),
`P(Y=1|theta)=logistic(a*(theta-b))`, D=1, theta~N(0,1). The equivalent binary
GRM has one intercept `d=-a*b`; recovered difficulty is `b=-d/a`.
No post-fit standardization or fitted affine alignment is permitted. The fixed
population variance and positive loading orientation identify the scale.

This is **a new library recovery regression using a published generating
model**, not replication of the paper's 3000-replication, 36-cell AIC model
selection experiment or of its sirt estimator.[1] The independent Cai et al.
(2011) two-group bifactor exact replication remains unimplemented: its
per-group factor masks, absent focal factor and uniform grid must not be
silently replaced. Neither task blocks private study calculations.

## Execution

```
RECOVERY_Q=121 cargo test -p mlsirm-core --test published_2pl_recovery \
  published_norm_gpu_parameter_recovery -- --ignored --exact --nocapture --test-threads=1
```

The test must be run under the installation's canonical exclusive GPU lease;
it does not acquire a project-local substitute. Repeat at an explicitly chosen
larger Q (e.g. 241) for integration sensitivity. No default quadrature value is
used by the GPU case. The portable simulator/scale test runs normally.

Simulation uses independently generated standard-normal abilities and Bernoulli
responses, seed 15110668; the seed and generator are test choices. No generating
parameter enters the fitter. The existing native fitter starts with slopes one
and empirical cumulative logits. Host initialization, quadrature construction,
reflection and metric calculation are not GPU optimization. The strict native
call must return convergence, terminal E dispatch, positive M/update counts,
and zero CPU numerical fallbacks. Precision/stopping are the native strict
f32 GPU implementation, marginal-loglik change bounded by
`1e-4 * (1 + abs(previous loglik))` after the native monotonicity check, at most 500 EM
iterations, four Newton iterations with documented explicit controls in source;
the ridge conditions the Hessian, not a Bayesian parameter prior.

All item estimates and signed errors are printed on the original a/b scales,
along with within-run mean error and RMSE, convergence, iterations, likelihood
and the same-call dispatch receipt. The prospective coarse regression bounds
(a RMSE <1; b RMSE <.35) detect gross scale/regression errors. They are
engineering guardrails, not published tolerances or an established scientific
accuracy criterion. Native errors fail the test rather than skipping it or
falling back to CPU.

**One fixed dataset is a smoke regression.** Across-item mean error is not
Monte Carlo bias; across-item RMSE is not per-parameter repeated-sampling RMSE.
A future MC acceptance must retain every attempted seed/failure, compute
per-parameter bias/RMSE/MCSE and convergence rates, and preregister accuracy
criteria. No MC, general bifactor, DIF or study scientific-adoption claim is
made by this test. Required canonical CI and independent review remain separate
from a local hardware execution.

## Sources

[1] https://www.mdpi.com/2078-2489/15/11/668 — Robitzsch, A. (2024). A note on equivalent and nonequivalent parametrizations of the two-parameter logistic item response model. *Information, 15*(11), 668. https://doi.org/10.3390/info15110668 (CC BY 4.0).
