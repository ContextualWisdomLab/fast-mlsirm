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

- No `.codegraph/` index exists in this repo today, so use normal search
  (grep/find, ripgrep) to locate and understand code. If a `.codegraph/` index
  is later added at the repo root, prefer CodeGraph
  (`codegraph explore "<query>"`, or the code-review-graph MCP tools) BEFORE
  grep/find — it surfaces callers/callees/impact that text search misses.

### This repo's role in the ecosystem

**fast-mlsirm** calibrates LLM-as-a-Judge outputs and manages the quality of the
measurement/evaluation items used in LLM-as-a-Judge; it incorporates aFIPC
Fixed-Item Parameter Calibration and kaefa item-fit-based optimal-model search.

It is one component of the ContextualWisdomLab ecosystem, which is organized
around **naruon** — the hub: an email/PIM system that DOM-decomposes emails and
files into a persisted knowledge graph. Each component is a **standalone program
that must ALSO work as a git submodule** (grown separately and together):
**waf-ids-ai-soc** (WAF/IDS/AI SOC/LB/APIM), **clearfolio** (document viewer),
**pg-erd-cloud** (ERD tool), **contextual-orchestrator** (LLM cost/perf/upstream-LB
gateway beyond LiteLLM), **codec-carver** (STT/omni-modal speech-video codec),
**fast-mlsirm** (this repo — LLM-as-a-Judge calibration + evaluation-item quality
via aFIPC FIPC + kaefa item-fit), **feelanet-adfs** (passwordless SSO —
OIDC/SCIM/ADFS/LDAP/FIDO2/OAuth2.1, eliminate passwords), **newsdom-api**
(PDF→DOM sidecar), and **semantic-data-portal** (upper-ontology/catalog/governance
plane with its own graph engine).

### Research grounding (attach paper PDFs)

Substantive feature/process PRs should find the relevant academic papers and
**commit their PDFs into the PR** (e.g. a `docs/papers/` or `references/`
directory) with full citations, respecting copyright: attach the PDF only when
redistribution is permissible; otherwise cite + link + summarize. For this repo,
that means IRT/psychometrics literature for aFIPC/kaefa item-fit calibration and
LLM-as-a-Judge evaluation methodology (see **Key Articles** above for the
existing MLSIRM/MLS2PLM reference set to build on).
<!-- END cwl-agent-guidance -->
