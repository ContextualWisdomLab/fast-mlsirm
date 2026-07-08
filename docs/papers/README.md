# Reference papers

Curated, in-repo copies of openly-licensed literature that grounds the
estimation core. Only papers under a license that permits redistribution are
stored here; everything else is referenced by DOI in [`../../AGENTS.md`](../../AGENTS.md).

## `wu2021-irt-stochastic-variational-inference.pdf`

Wu, M., Davis, R. L., Domingue, B. W., Piech, C., & Goodman, N. (2021).
*Modeling Item Response Theory with Stochastic Variational Inference.*
arXiv:2108.11579. https://arxiv.org/abs/2108.11579

- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/. Redistributed here unmodified
  with attribution, as the license permits.
- **Why it is here:** it develops a *fast, scalable* estimator for item
  response theory by mapping the likelihood and its gradient onto
  data-parallel, accelerator-friendly computation (amortized/stochastic
  variational inference). That is exactly the numerical shape this project
  accelerates: the penalized negative log-likelihood and gradient hot path of
  the MLSIRM/MLS2PLM family, now offloadable to the GPU via the wgpu GPGPU
  kernels in `crates/mlsirm-core/src/gpu.rs`. The paper is the design reference
  for keeping GPU-accelerated IRT estimation numerically faithful to the CPU
  objective.
