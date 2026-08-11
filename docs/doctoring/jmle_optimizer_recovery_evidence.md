# JMLE Rust optimizer recovery evidence

## Purpose

PR #760 moved the result-affecting Adam, L-BFGS, and `adam_lbfgs` optimizer loops for public `backend="rust"` JMLE into `mlsirm-core`. Delegation and parity tests establish numerical ownership, but ownership alone does not establish that each public optimizer mode can recover a known generating scale. Issue #626 therefore retains a separate scientific acceptance gate.

`tests/test_jmle_optimizer_recovery.py` adds deterministic public-surface evidence using one correctly specified unidimensional 2PL-generating sample (`gamma=0`) and the public MIRT/JMLE fit path. Each advertised optimizer is evaluated on the same data and must use the Rust CPU path, report convergence, reduce the objective, and satisfy explicit finite-sample bias, MAE, and RMSE bounds for item discrimination, item easiness, and standardized person ability.

## Interpretation boundary

This is optimizer-mode recovery evidence, not a claim that penalized joint maximum likelihood is asymptotically unbiased for item parameters. JMLE retains the incidental-parameter limitations of joint estimation. The test therefore uses deliberately broad acceptance bounds and reports multiple error summaries rather than correlation-only evidence. Marginal maximum likelihood remains the more appropriate consistency target for item-parameter recovery claims when its model assumptions apply.

The study does not authorize an end-to-end GPU optimizer claim. Existing GPU objective/gradient parity remains distinct from optimizer-state parity; profiling and dedicated parity/recovery evidence are required before such a path is advertised.

## Evidence contract

For each of `adam`, `lbfgs`, and `adam_lbfgs`:

- exact public `fit(..., estimator="jmle", backend="rust", rust_device="cpu")` is exercised;
- convergence status and positive iteration count are required;
- the objective trace must remain finite and finish no worse than it starts;
- discrimination and easiness expose bias, MAE, and RMSE gates;
- person ability exposes standardized MAE and RMSE gates to respect the identified latent scale; and
- failure messages carry the complete metric dictionary so CI evidence is diagnostically useful rather than a binary pass/fail.

## References

Harwell, M., Stone, C. A., Hsu, T.-C., & Kirisci, L. (1996). Monte Carlo studies in item response theory. *Applied Psychological Measurement, 20*(2), 101–125. https://doi.org/10.1177/014662169602000201

Reckase, M. D. (2009). *Multidimensional item response theory*. Springer. https://doi.org/10.1007/978-0-387-89976-3

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations*. https://arxiv.org/abs/1412.6980

Liu, D. C., & Nocedal, J. (1989). On the limited memory BFGS method for large scale optimization. *Mathematical Programming, 45*, 503–528. https://doi.org/10.1007/BF01589116
