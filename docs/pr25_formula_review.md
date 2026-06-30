# PR #25 Formula Review

Review target: https://github.com/ContextualWisdomLab/fast-mlsirm/pull/25

## Verdict

The formula refactor in PR #25 is mathematically valid for the likelihood and
gradient contract implemented by `fast-mlsirm`. The proposed `grad_alpha` and
`grad_theta` changes are algebraic rewrites of the existing sums, not a change
to the MLS2PLM model.

The PR should still be refreshed before merge because GitHub reports it as not
mergeable against the current `main`, and the merge conflict is real. In
particular, `main` already contains a one-hot projection form for `grad_theta`,
while PR #25 proposes an equivalent boolean-mask projection form.

## Paper And Formula Basis

The MLS2PLM paper defines the response model as a multidimensional two-parameter
logistic item response model augmented by a latent-space distance term:

```text
logit P(Y_pi = 1) = alpha_i + beta_i * theta_p,d(i) - gamma * ||zeta_p - zeta_i||
```

The local project contract uses equivalent parameter names and positive
reparameterizations:

```text
eta_pi = exp(alpha_i) * theta_p,d(i) + b_i - exp(tau) * r_pi
r_pi = sqrt(sum_k (xi_pk - zeta_ik)^2 + eps)
loss_pi = softplus(eta_pi) - y_pi * eta_pi
e_pi = sigmoid(eta_pi) - y_pi
```

Sources checked:

- Kang, I., & Jeon, M. (2025), "Multidimensional Latent Space Item
  Response Models: A Note on the Relativity of Conditional Dependence",
  Psychometrika 90(2), 799-826 (`doi:10.1017/psy.2025.5`).
- Jeon, M., Jin, I. H., Schweinberger, M., & Baugh, S. (2021),
  "Mapping Unobserved Item-Respondent Interactions: A Latent Space Item Response
  Model with Interaction Map" (`doi:10.1007/s11336-021-09762-5`).
- Psychometrika article page for
  "Multidimensional Latent Space Item Response Models: A Note on the Relativity
  of Conditional Dependence":
  https://www.cambridge.org/core/journals/psychometrika/article/multidimensional-latent-space-item-response-models-a-note-on-the-relativity-of-conditional-dependence/7F70D92C0A90660962C361F01E462C40
- Local formula contract in `docs/prd_trd_summary.md`.

## Implementation Check

The current `main` implementation uses:

```python
grad_alpha = (e * a[None, :] * params.theta[:, factors]).sum(axis=0)
I = np.zeros((e.shape[1], params.theta.shape[1]), dtype=e.dtype)
I[np.arange(e.shape[1]), factors] = 1
grad_theta = (e * a[None, :]) @ I
```

PR #25 proposes:

```python
grad_alpha = a * np.sum(e * params.theta[:, factors], axis=0)
mask = (factors[:, None] == np.arange(params.theta.shape[1])).astype(e.dtype)
mask *= a[:, None]
grad_theta = e @ mask
```

These are equivalent because:

```text
d loss / d alpha_i = sum_p e_pi * exp(alpha_i) * theta_p,d(i)
d loss / d theta_pd = sum_{i: d(i)=d} e_pi * exp(alpha_i)
```

Pulling `exp(alpha_i)` outside the person sum preserves `grad_alpha`, and the
mask matrix is the same item-to-dimension indicator matrix as `I`, scaled by
`a_i`.

The distance gradients already present in `main` are consistent with the same
NLL contract:

```text
d loss / d xi_pk = -sum_i e_pi * gamma * (xi_pk - zeta_ik) / r_pi
d loss / d zeta_ik =  sum_p e_pi * gamma * (xi_pk - zeta_ik) / r_pi
d loss / d tau    =  sum_pi e_pi * (-exp(tau) * r_pi)
```

## Verification Run

Environment: Windows PowerShell, Python 3.12.3.

Commands run:

```text
py -3 -m pytest
py -3 -m pytest  # in detached PR #25 worktree
custom main-vs-PR objective/gradient equivalence script
cargo test
```

Results:

- `main`: `36 passed`.
- PR #25 worktree: `12 passed`.
- Main-vs-PR objective/gradient comparison passed for `MIRT`, `MLSRM`,
  `MLS2PLM`, `ULSRM`, and `ULS2PLM` with `1e-12` absolute tolerance.
- `cargo test` could not be run because `cargo` is not installed or not on PATH
  in this environment.

## Merge Recommendation

Approve the formula refactor after refreshing PR #25 against current `main`.
The mathematical change is sound, but the branch should not merge in its current
state because:

- GitHub/OpenCode reports merge conflicts.
- The PR includes unrelated formatting and a committed `.coverage` database.
- Current `main` already contains a mathematically equivalent `grad_theta`
  vectorization, so conflict resolution should choose one clear projection
  implementation and avoid keeping both historical comments.
