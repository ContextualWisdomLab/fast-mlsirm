# MLS2PLM canonical equations (source of truth for the numeric core)

This note pins the exact equations that `fast-mlsirm` implements, so that any
future performance refactor or "optimization" of the numeric core can be checked
against the published formula rather than against a previous (possibly already
drifted) version of the code.

The canonical implementation is a **simple-structure specialization** of the
Multidimensional Latent Space Item Response Model. See
`docs/pr25_formula_review.md` for why the simple-structure restriction is a
model choice, not an implementation detail that may be silently rewritten into
the full discrimination-vector model.

## Source papers

- Jeon, M., Jin, I. H., Schweinberger, M., & Baugh, S. (2021). *Mapping
  Unobserved Item-Respondent Interactions: A Latent Space Item Response Model
  with Interaction Map.* Psychometrika, 86(2), 378-403.
  DOI: [10.1007/s11336-021-09762-5](https://doi.org/10.1007/s11336-021-09762-5)
  — defines the Latent Space Item Response Model (LSIRM):
  `logit P(Y = 1) = theta_p + b_i - gamma * ||u_p - v_i||`.
- Molenaar, D., & Jeon, M. (2026). *Regularized Joint Maximum Likelihood
  Estimation of Latent Space Item Response Models.* Psychometrika, 91, 335-359.
  DOI: [10.1017/psy.2025.10068](https://doi.org/10.1017/psy.2025.10068)
  — regularized (penalized) joint maximum-likelihood estimation, which is the
  estimation strategy this package uses (L2 penalties on the parameter blocks).
- Kang, I., & Jeon, M. (2025). *Multidimensional Latent Space Item Response
  Models: A Note on the Relativity of Conditional Dependence.* Psychometrika,
  90(2), 799-826. DOI: [10.1017/psy.2025.5](https://doi.org/10.1017/psy.2025.5)
  — the multidimensional (MLS2PLM) generalization.

PDFs are not redistributed here (the primary sources are not open-access); cite
via the DOIs above. This document reproduces only the equations needed to verify
the code, which are facts and not copyrightable expression.

## Model

For person `p`, item `i` with `factor_id(i) = d(i)`, latent trait vector
`theta_p`, person latent position `xi_p` and item latent position `zeta_i` in a
shared latent space:

```text
r_pi  = sqrt( || xi_p - zeta_i ||^2 + eps_distance )          (regularized distance)
eta_pi = a_i * theta_{p, d(i)} + b_i - gamma * r_pi           (linear predictor)
P(Y_pi = 1) = sigmoid(eta_pi)
```

with the positivity reparameterizations

```text
a_i   = exp(alpha_i)        (discrimination, > 0)
gamma = exp(tau)            (interaction weight, > 0)
```

`eps_distance` (default `1e-8`) smooths the Euclidean distance so that `r_pi`
and its gradient are finite when `xi_p = zeta_i`. It is applied **inside** the
square root in both the forward pass and the gradient, keeping `d r_pi / d xi_p
= (xi_p - zeta_i) / r_pi` exactly consistent.

## Objective (penalized negative log-likelihood)

Over observed entries `O` (mask), with Bernoulli responses `y_pi in {0, 1}`:

```text
NLL_data = sum_{(p,i) in O} [ softplus(eta_pi) - y_pi * eta_pi ]
```

This is the standard binary cross-entropy written in a numerically stable form:
`softplus(eta) - y*eta = -[ y*log P + (1-y)*log(1-P) ]`. The reported
log-likelihood is `loglik = -NLL_data` (data term only, before the penalty).

The regularized objective adds L2 penalties (Molenaar & Jeon, 2026):

```text
NLL = NLL_data
    + 0.5 * lambda_theta * ||theta||^2
    + 0.5 * lambda_b     * ||b||^2
    + 0.5 * lambda_alpha * ||alpha - mu_alpha||^2      (only if a_i is free)
    + 0.5 * lambda_xi    * ||xi||^2                    (only if uses_space)
    + 0.5 * lambda_zeta  * ||zeta||^2                  (only if uses_space)
    + 0.5 * lambda_tau   * (tau - mu_tau)^2            (only if uses_space)
```

(`uses_space` is true for all models except `MIRT`; see "Model variants".)

## Gradient

Let `e_pi = (P(Y_pi = 1) - y_pi)` on observed entries (0 elsewhere). The
data-term gradients are:

```text
d NLL / d b_i        = sum_p e_pi
d NLL / d alpha_i    = sum_p e_pi * a_i * theta_{p, d(i)}   # da_i/dalpha_i = a_i
d NLL / d theta_{p,k}= sum_{i : d(i)=k} e_pi * a_i
d NLL / d xi_p       = -gamma * sum_i e_pi * (xi_p - zeta_i) / r_pi
d NLL / d zeta_i     =  gamma * sum_p e_pi * (xi_p - zeta_i) / r_pi
d NLL / d tau        = sum_{(p,i)} e_pi * (-gamma * r_pi)   # dgamma/dtau = gamma
```

The `alpha` and `tau` gradients carry the chain-rule factors from the
reparameterizations `a_i = exp(alpha_i)` and `gamma = exp(tau)`.

The L2 penalty adds `lambda_* * (param - center)` to each block (`center = 0`
except `mu_alpha`, `mu_tau`).

## Model variants

`model_flags(model)` toggles two switches:

- `free_alpha` is false for `MLSRM` / `ULSRM` (Rasch-type: `a_i = 1`, so
  `alpha` has no gradient contribution).
- `uses_space` is false for `MIRT` (no latent space: `gamma = 0`, `r_pi = 0`,
  and `xi`, `zeta`, `tau` have no gradient contribution).

## Where this is enforced in code

- `python/fast_mlsirm/objective.py` — `linear_predictor`, `neg_loglik_and_grad`
  (NumPy reference path) and `_add_penalty`.
- `crates/mlsirm-core/src/lib.rs` — `neg_loglik_and_grad` (Rust path); kept at
  parity with the NumPy path by `tests/test_objective.py`.
- `tests/test_objective.py` — `test_neg_loglik_matches_closed_form_single_entry`
  and `test_neg_loglik_and_grad_matches_independent_reference` pin the objective
  and every gradient block to the equations above, computed independently with
  the Python standard library (no NumPy vectorization shared with production).
