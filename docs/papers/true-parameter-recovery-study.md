# Literature-validated true-parameter recovery study

## Scope

This study is a Rust-only regression experiment for the repository's existing
simple-structure MLS2PLM contract. It does **not** change the model
parameterization, likelihood, gradients, priors, or identification rules.
Python is not used to generate responses, fit parameters, align latent
positions, or calculate recovery statistics.

The merge-gate experiment uses a bounded, representative condition from Kang
and Jeon (2025): two trait dimensions, eight items per dimension, a
two-dimensional interaction map, and `P = 500`, one of the paper's three
sample-size conditions. The dimensionality and item-count settings are the
paper's lowest levels, while `P = 500` is deliberately retained as a stable CI
recovery sentinel rather than described as the minimum sample-size condition.
The full paper uses 50 replications per condition; CI uses one deterministic
sentinel replication so that every pull request remains bounded and
reproducible. The long-running ignored-test jobs execute the sentinel together
with all other repository-owned statistical studies.

## Verified response equation

Kang and Jeon (2025, Equation 3) define the multidimensional latent-space 2PL
model as

\[
\operatorname{logit}\Pr(Y_{pi}=1)
  = \mathbf a_i^\mathsf T\boldsymbol\theta_p + b_i
    - \gamma\,d(\boldsymbol\xi_p,\boldsymbol\zeta_i).
\]

Under the paper's between-item simple-structure restriction, item `i` loads on
one trait dimension `d(i)`, so

\[
\mathbf a_i^\mathsf T\boldsymbol\theta_p
  = a_i\theta_{p,d(i)}.
\]

The repository stores `alpha_i = log(a_i)` and `tau = log(gamma)`. Its exact
implemented equation is therefore

\[
\eta_{pi}
  = \exp(\alpha_i)\theta_{p,d(i)} + b_i
    - \exp(\tau)\sqrt{\sum_k(\xi_{pk}-\zeta_{ik})^2+\varepsilon}.
\]

This is algebraically the paper model under simple structure, with the small
`epsilon` term used only to keep the Euclidean-distance derivative finite at
coincident points. The experiment generates data with this same sign
convention: `b_i` is the item intercept/easiness parameter, not positive item
difficulty.

## Simulation cell

The deterministic Rust experiment follows a bounded paper condition with the
lowest dimensionality and item-count levels from Kang and Jeon (2025):

- persons: `P = 500` (one of the paper's `300`, `500`, and `1000` conditions),
- trait dimensions: `D = 2`,
- items per dimension: `I_d = 8`,
- latent-space dimensions: `K = 2`,
- discrimination values: evenly spaced over `[0.5, 2.5]`,
- intercept/easiness values: evenly spaced over `[0, 5]` and deterministically
  permuted,
- trait correlation: `rho = 0.30`,
- person and item interaction coordinates: standard bivariate normal,
- distance weight: `gamma = 1.5`,
- binary responses: Bernoulli draws from `sigmoid(eta_pi)`.

Kang and Jeon evaluate recovery with mean squared error, absolute bias, and
sampling variability after resolving latent-map indeterminacy. Molenaar and
Jeon (2026) likewise compare regularized estimators after rotation. The CI
sentinel records correlation and RMSE for invariant or identified quantities:

- item discrimination `a_i`,
- interaction-adjusted easiness
  `b_i - gamma E_xi[d(xi, zeta_i)]`,
- EAP trait scores by dimension,
- pairwise item-map distances, invariant to translation, rotation, and
  reflection,
- the positive distance weight `gamma`,
- monotonicity and finiteness of the marginal log-likelihood trace.

The adjusted-easiness statistic is used because raw intercept and item-map
radius are partially confounded in a distance model. Pairwise distances provide
an orientation-free map metric; this serves the same identification purpose as
the Procrustes alignment used in the source simulation study.

## Execution evidence

- **Rust backend:** response generation, MMLE-EM fitting, and every recovery
  statistic are implemented in the Rust integration test.
- **CPU:** the dedicated recovery job runs the Rust experiment with
  `Device::Cpu`. The core objective uses coarse person shards through
  `std::thread::scope` for samples at or above its multithreading threshold.
- **GPU:** an Ubuntu job installs Mesa's Lavapipe Vulkan driver, executes the
  explicit `rust_device="gpu"` parity test, rejects any skipped result, and runs
  the Rust recovery device-parity experiment.
- **Ignored tests:** CI executes all repository-owned ignored Rust studies. The
  CPU recovery sentinel and the explicit GPU recovery test are run in dedicated
  jobs; the remaining ignored workspace and excluded-PyO3 tests are swept
  separately so no ignored path is silently omitted.
- **Coverage and docstrings:** the repository thresholds remain fixed at 100%
  in `pyproject.toml` and the Cargo workspace metadata. This PR adds no Python
  production API and therefore creates no uncovered Python or undocumented
  public surface.

## Primary sources

1. Kang, I., & Jeon, M. (2025). Multidimensional Latent Space Item Response
   Models: A Note on the Relativity of Conditional Dependence. *Psychometrika,
   90*(2), 799-826. https://doi.org/10.1017/psy.2025.5
2. Jeon, M., Jin, I. H., Schweinberger, M., & Baugh, S. (2021). Mapping
   Unobserved Item-Respondent Interactions: A Latent Space Item Response Model
   with Interaction Map. *Psychometrika, 86*(2), 378-403.
   https://doi.org/10.1007/s11336-021-09762-5
3. Molenaar, D., & Jeon, M. (2026). Regularized Joint Maximum Likelihood
   Estimation of Latent Space Item Response Models. *Psychometrika, 91*,
   335-359. https://doi.org/10.1017/psy.2025.10068

The Cambridge articles are openly accessible. No publisher PDF is committed;
the citations, DOI links, equation traceability, and implementation summary are
included instead.
