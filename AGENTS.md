# Repository Guidance

## Paper-First Research

Before changing model formulas, fit diagnostics, estimators, simulation
contracts, or interpretation-facing outputs, research the relevant MLSIRM,
MLS2PLM, and psychometric fit-statistic literature first. Summarize the paper
basis in the plan before editing code, and keep the implementation inside the
paper-supported scope unless the task explicitly asks for a new model-design
PR.

## Key Articles

Start with these references before model or fit-diagnostic work:

- Kang, I., & Jeon, M. (2025). "Multidimensional Latent Space Item
  Response Models: A Note on the Relativity of Conditional Dependence."
  Psychometrika, 90(2), 799-826. doi:10.1017/psy.2025.5.
- Jeon, M., Jin, I. H., Schweinberger, M., & Baugh, S. (2021). "Mapping
  Unobserved Item-Respondent Interactions: A Latent Space Item Response
  Model with Interaction Map." Psychometrika, 86(2), 378-403.
  doi:10.1007/s11336-021-09762-5.
- Molenaar, D., & Jeon, M. (2026). "Regularized Joint Maximum Likelihood
  Estimation of Latent Space Item Response Models." Psychometrika, 91,
  335-359. doi:10.1017/psy.2025.10068.
- Tay, L., Ali, U. S., Drasgow, F., & Williams, B. (2011). "Fitting IRT
  Models to Dichotomous and Polytomous Data: Assessing the Relative
  Model-Data Fit of Ideal Point and Dominance Models." Applied
  Psychological Measurement, 35(4), 280-295. doi:10.1177/0146621610390674.
- Roberts, J. S., Donoghue, J. R., & Laughlin, J. E. (1998). "The
  Generalized Graded Unfolding Model: A General Parametric Item Response
  Model for Unfolding Graded Responses." ETS Research Report Series.
  doi:10.1002/j.2333-8504.1998.tb01781.x.
- Orlando, M., & Thissen, D. (2000). "Likelihood-Based Item-Fit Indices
  for Dichotomous Item Response Theory Models." Applied Psychological
  Measurement, 24, 50-64.
- Maydeu-Olivares, A., & Joe, H. (2005). "Limited- and Full-Information
  Estimation and Goodness-of-Fit Testing in 2^n Contingency Tables."
  Journal of the American Statistical Association, 100(471), 1009-1020.
  doi:10.1198/016214504000002069.
- Drasgow, F., Levine, M. V., & Williams, E. A. (1985). "Appropriateness
  Measurement with Polychotomous Item Response Models and Standardized
  Indices." British Journal of Mathematical and Statistical Psychology,
  38(1), 67-86. doi:10.1111/j.2044-8317.1985.tb00817.x.
- Fox, J.-P., & Glas, C. A. W. (2001). "Bayesian Estimation of a
  Multilevel IRT Model." Psychometrika, 66, 271-288.
  doi:10.1007/BF02294839.
- Bock, R. D., & Zimowski, M. F. (1997). "Multiple Group IRT." In W. J.
  van der Linden & R. K. Hambleton (Eds.), Handbook of Modern Item
  Response Theory.
- Chalmers, R. P. (2012). "mirt: A Multidimensional Item Response Theory
  Package for the R Environment." Journal of Statistical Software, 48(6).
  doi:10.18637/jss.v048.i06.

## Formula Scope

Treat the current Python and Rust formulas as a valid simple-structure
specialization of the MLS2PLM paper, not as the full general discrimination-
vector MLS2PLM model.

The current local contract is:

```text
eta_pi = exp(alpha_i) * theta_p,d(i) + b_i - exp(tau) * r_pi
r_pi = sqrt(sum_k (xi_pk - zeta_ik)^2 + eps)
```

The original multidimensional paper writes the response term as:

```text
logit P(Y_pi = 1) = a_i^T theta_p + b_i - gamma * d(xi_p, zeta_i)
```

The implementation formula matches the original MLS2PLM formula under the
simple-structure restriction `a_i^T theta_p = a_i * theta_p,d(i)`. Do not merge
piecemeal PRs that attempt to "fix", "renovate", or reinterpret the formula
through local gradient, distance, masking, or vectorization edits. Those
attempts are not actionable unless they are part of an explicit model-design PR
that updates the parameterization, likelihood, analytic gradients, tests, docs,
and Rust parity together.

Close formula-renovation attempts that only modify local algebra or performance
plumbing while leaving the model contract ambiguous.

If full MLS2PLM support is desired, implement it as a separate complete model
path instead of mutating the existing simple-structure formula in place. That
work should update parameter shapes, simulation, likelihood, analytic gradients,
tests, documentation, and Rust parity together.

## Local build

- **Editable install is deterministic with Rust on `PATH`.** The editable
  install compiles `fast_mlsirm._core` with maturin, so keeping `cargo`/`rustc`
  discoverable avoids network-dependent bootstrap during PEP 517 builds. If
  cargo is absent, maturin may try to provision a temporary Rust toolchain via
  `puccinialin`; set `MATURIN_NO_INSTALL_RUST=1` when you need a fail-fast
  offline/proxy-safe build. A proxy or certificate error in that fallback is not
  proof of a Python/PyO3 incompatibility. Prefer an explicit Rust toolchain on
  `PATH` (e.g. Windows git-bash: `export PATH="$HOME/.cargo/bin:$PATH"`), then
  `pip install -e .[dev]`. No manual `maturin develop` step is needed.
- **Python support claims must match evidence.** `pyproject.toml` declares
  `requires-python = ">=3.12"`, matching the required CI matrix legs on CPython
  3.12 and 3.14. Do not advertise a lower floor than the hashed CI dependency
  lock can install.

<!-- BEGIN cwl-agent-guidance -->
## Agent guidance (CWL governance)

Guidance for ANY agent (Claude, Codex, Cursor, opencode, ...) working in this repo.

### Security & review gate

- Every PR runs a central **Security Scan** required gate: `osv-scan` +
  `dependency-review` (diff-scoped) and `trivy-fs` (repo-wide, CRITICAL/HIGH,
  fixable only). It runs against every PR base, **including stacked PRs**.
- Here that surface is dependency manifests/lockfiles: **`Cargo.lock`** and the
  workspace crates (`crates/mlsirm-core`, `crates/fast-mlsirm-py`) for Rust, and
  **`pyproject.toml`** (maturin/pyo3 build, `numpy` runtime dep) for Python.
  There is no Dockerfile or k8s manifest, so expect findings to point at a
  vulnerable crate or Python dependency.
- A **failing `trivy-fs` is a REAL finding, not a flake.** Read the job log (it
  prints each finding's rule id / severity / file) or the run's SARIF results,
  then **remediate**: bump the offending crate (`cargo update -p <crate>`,
  refresh `Cargo.lock`) or the Python dependency in `pyproject.toml`. Only for a
  genuine false positive, add a narrow, documented `.trivyignore(.yaml)` entry.
  Do NOT weaken or disable the gate.
- Reproduce locally against the **merge ref**, not just the PR head, and refresh
  the DB first: `trivy --download-db-only` then `trivy fs .` (a stale DB misses
  findings).
- The org `code_scanning` ruleset is intentionally **CodeQL-only** (multiple
  code-scanning tools can't converge on one PR ref). Gating is by the Security
  Scan **job result**, not the `code_scanning` rule — don't add tools to that rule.

### Code exploration

- Use CodeGraph before grep/find, ripgrep, or broad file reads whenever code
  needs to be located or understood. If the repository root does not yet have
  a `.codegraph/` index, initialize it with `codegraph init .` first, then use
  `codegraph explore "<query>"` (or the code-review-graph MCP tools). Keep the
  index refreshed for the current checkout so callers, callees, and impact are
  derived from the active PR head rather than stale source.

### This repo's role in the ecosystem

**fast-mlsirm** is the reusable, domain-neutral measurement and psychometric
computation layer. It owns versioned AssessmentSpec/RubricSpecification/Scoring
contracts, item/rater observations, calibration, model diagnostics, linking,
DIF/invariance/fairness, factor/model selection, recovery/simulation, and the
Rust-first numerical kernels that implement those contracts.

`ContextualWisdomLab/psychometrics-commons` is a **downstream consumer** and the
canonical hosted product repository. It owns product HTTP/admin APIs,
participant/session/consent/result lifecycle, product persistence/migrations,
resource authorization, reference clients, research-release orchestration, and
deployment composition. fast-mlsirm must remain independently installable and
must never depend on Psychometrics Commons product code, ORM/database models,
HTTP types, UI code, or deployment configuration.

Cross-repository integration uses explicit versioned contracts or immutable
artifacts; no product-specific field belongs in fast-mlsirm unless it is
reusable across independent assessment domains. The hosted product must not be
recreated under `services/assessment_runtime` in this repository.

`kaefa`, `aFIPC`, and `nonnest2` are not runtime, build, CI-oracle, or release dependencies
for fast-mlsirm or Psychometrics Commons. Methods needed from the literature are
implemented in fast-mlsirm from primary methodological sources and independent
numerical/recovery evidence rather than by embedding those legacy R packages or
using them as the sole validation oracle.

fast-mlsirm remains one independently usable component of the broader
ContextualWisdomLab ecosystem. Other repositories own their own bounded
contexts, including Keyverse for identity/federation, TEPP for temporal/event
analysis, Gyeot for EMA/ESM collection, semantic-data-portal for research
catalog/release provenance, contextual-orchestrator for bounded LLM
orchestration, and EgressWeave for controlled external egress. Treat those as
integrations, not hidden implementation dependencies.

### Temporal/event and Context Fabric ownership

TEPP owns temporal/event composition and semantics, including event ontology,
valid/system/event-time meaning, event ordering, changing-membership history,
and longitudinal leakage policy. fast-mlsirm owns reusable time-indexed
psychometric numerical kernels over explicit supplied occasion/time carriers.
A TEPP-originated temporal design enters this repository only through a
versioned immutable Anti-Corruption Layer; cross-service SQL, direct TEPP
database access, and hidden TEPP runtime dependencies are prohibited.

`ContextualWisdomLab/context-graph-contracts` is the contract-only Shared Kernel
for Context Assertions, CloudEvents, provenance, conformance, and admission;
`ContextualWisdomLab/enterprise-architecture-core` is the authoritative EA
Decision Plane. Treat both as foreign owner surfaces from this repository.
Architecture/package/backend/toolchain/consumer-lifecycle facts may be projected
only through an immutable released `context-graph-contracts` contract with
provenance. Estimator values, latent scores, DIF/fit diagnostics, and
scientific-validity evidence are not authoritative EA facts. Do not pin an
unreleased sibling PR head, duplicate those facts into EA authority, or modify
Context Fabric repositories as a leaf workaround.

### Research grounding (attach paper PDFs)

Substantive feature/process PRs should find the relevant academic papers and
**commit their PDFs into the PR** (e.g. a `docs/papers/` or `references/`
directory) with full citations, respecting copyright: attach the PDF only when
redistribution is permissible; otherwise cite + link + summarize. For this repo,
that means the primary IRT/psychometrics and LLM-as-a-Judge measurement
literature governing the specific method being implemented (see **Key Articles**
above for the existing MLSIRM/MLS2PLM reference set to build on). Do not treat a
legacy package implementation as a substitute for primary-source validation.
<!-- END cwl-agent-guidance -->

## Code-owner review gates — disabled (on hold)

As of 2026-08-04, code-owner review requirements (`require_code_owner_reviews` in branch
protection, `require_code_owner_review` in rulesets) are disabled across the ContextualWisdomLab
org: there is a single maintainer (solo developer), so a code-owner approval gate can never be
satisfied. This is ON HOLD until the org has multiple maintainers — do NOT re-enable these
settings or add CODEOWNERS-based merge gates before then.
