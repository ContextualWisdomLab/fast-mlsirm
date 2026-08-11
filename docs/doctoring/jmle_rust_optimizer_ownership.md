# Doctoring: Rust-owned JMLE optimizers

## Claim

When `FitConfig(backend="rust")` is selected, public joint maximum likelihood
estimation (JMLE) executes Adam moment updates, L-BFGS two-loop recursion with
Armijo line search, and `adam` / `lbfgs` / `adam_lbfgs` sequencing in the
compiled Rust core. Python retains validation, packed-vector marshalling, random
restarts, and result packaging only.

## Standards and literature (APA 7th)

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In
*3rd International Conference on Learning Representations (ICLR)*.
https://arxiv.org/abs/1412.6980

Liu, D. C., & Nocedal, J. (1989). On the limited memory BFGS method for large
scale optimization. *Mathematical Programming, 45*(1–3), 503–528.
https://doi.org/10.1007/BF01589116

Nocedal, J., & Wright, S. J. (2006). *Numerical optimization* (2nd ed.). Springer.
https://doi.org/10.1007/978-0-387-40065-5

## Verification

- Rust unit tests for quadratic recovery under Adam, L-BFGS, and sequenced
  `adam_lbfgs`.
- Python ownership sentinel that monkeypatches legacy `_adam` / `_lbfgs` and
  requires `backend="rust"` public `fit` to avoid those loops.
- Existing `tests/test_fit_pipeline.py` smoke coverage for rust-backed JMLE.
