# fast-mlsirm PRD/TRD Summary

## Product Goal

`fast-mlsirm` provides fast simulation, fitting, and recovery diagnostics for
Multidimensional Latent Space Item Response Models, especially MLS2PLM:

```text
logit P(Y_pi = 1) = a_i * theta_p,d(i) + b_i - gamma * distance(xi_p, zeta_i)
```

The package is aimed at psychometrics, educational measurement, mental-health
assessment, item diagnostics, adaptive testing research, and production-scale
binary response scoring pipelines.

For sale and support purposes, the current product is a commercial beta for
technical users. It can be packaged, verified, and supported for the documented
local API/CLI workflows, but it is not a regulated decision product, hosted
platform, or full ordinal/Bayesian estimation system.

## MVP Scope

Must have:

- Canonical MLS2PLM simulation.
- `gamma=0` no-CD simulation.
- `MIRT`, `MLSRM`, and `MLS2PLM` model constraints.
- Missing response exclusion.
- Likelihood and analytic gradient.
- Adam and L-BFGS-style optimizers.
- Procrustes alignment and recovery reports.
- Python API and CLI.
- Rust core formulas for likelihood and gradient.
- Optional PyO3/maturin binding for using the Rust likelihood and gradient from
  Python fitting.

Explicitly out of MVP:

- Full HMC/NUTS Bayesian sampling.
- Ordinal graded response models.
- Real-time adaptive testing.
- GUI dashboards.
- Automatic psychological construct naming.

## Architecture

The intended architecture is Python API first, Rust numerical core second:

```text
python/fast_mlsirm/
  config, simulation, objective, fit, diagnostics, cli

crates/mlsirm-core/
  model structs, stable likelihood, analytic gradients, Rust tests

crates/fast-mlsirm-py/
  PyO3 module exposed as fast_mlsirm._core
```

The default Python backend is vectorized NumPy. The optional Rust backend uses
the same core formula through a PyO3/maturin extension and can be selected with
`FitConfig(backend="rust")`, `FitConfig(backend="auto")`, or
`fast-mlsirm fit --backend`. Source and editable installs build that extension
with maturin and therefore require a Rust toolchain; NumPy remains the default
runtime backend after installation. The PyO3 crate is built through maturin and
validated through Python backend parity tests, while `cargo test --workspace`
covers the standalone Rust core.

## Formula Contract

For item `i` assigned to factor `d_i`:

```text
eta_pi = exp(alpha_i) * theta_p,d_i + b_i - exp(tau) * r_pi
r_pi = sqrt(sum_k (xi_pk - zeta_ik)^2 + eps)
loss = softplus(eta_pi) - y_pi * eta_pi
```

The NLL gradient uses:

```text
e_pi = sigmoid(eta_pi) - y_pi
```

and applies L2 regularization to `theta`, `xi`, `zeta`, `b`, `alpha`, and
`tau` where those parameters are active for the selected model.

## Roadmap

1. Stabilize Python reference formulas and tests.
2. Maintain NumPy/Rust objective parity through PyO3/maturin tests.
3. Add block-mode likelihood/gradient execution.
4. Add benchmark harness and repeated recovery-grid runner.
5. Add sparse/missing optimized kernels.
6. Explore JAX/GPU and ordinal response extensions as separate model/runtime
   design work.
