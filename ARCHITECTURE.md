# Architecture — fast-mlsirm

Status: living baseline (commercial-hardening loop)  
Audience: implementers, reviewers, buyers performing technical due diligence  
Related: `docs/prd_trd_summary.md`, `docs/mmle_marginal_lsirm_design.md`, `AGENTS.md`, `CLAUDE.md`, `docs/doctoring/`

## 1. Purpose

`fast-mlsirm` is a psychometrics / educational measurement library for
**Multidimensional Latent Space Item Response Models** (MLSIRM / MLS2PLM) and
related IRT tooling. It is designed to:

1. **Stand alone** as a local Python package with a Rust numeric core.
2. **Compose** as a module inside the ContextualWisdomLab ecosystem
   (central `.github` governance, `naruon` product surfaces, optional
   `contextual-orchestrator` LLM workflows) without requiring those siblings
   at import time.
3. Prefer **paper-backed** estimators and **true-parameter recovery** evidence
   over keyword heuristics or demo stubs.

## 2. Layered system view

```text
┌──────────────────────────────────────────────────────────────────────┐
│ Presentation / product surfaces (optional)                           │
│  HTML diagnostics reports · CLI · enterprise sales readiness scripts │
├──────────────────────────────────────────────────────────────────────┤
│ Python orchestration (python/fast_mlsirm/)                           │
│  fit API · config validation · simulation · I/O · scoring adapters   │
│  multilevel/longitudinal *contracts* (when merged) · CAT / DIF / etc │
├──────────────────────────────────────────────────────────────────────┤
│ PyO3 boundary (crates/fast-mlsirm-py → fast_mlsirm._core)            │
├──────────────────────────────────────────────────────────────────────┤
│ Rust numeric core (crates/mlsirm-core)  ★ hot path                    │
│  likelihood + analytic gradients · MMLE / multigroup / multilevel    │
│  GPU marginal path (wgpu) · CPU multithreaded rayons-style work      │
│  recovery-contract tests · fuzz targets                              │
└──────────────────────────────────────────────────────────────────────┘
```

**Rule:** pure numeric work lives in Rust. Python owns validation, packaging,
report rendering, and user-facing contracts. GPU device selection is explicit;
CPU remains the default portable path.

## 3. Modular MSA stance (standalone + embeddable)

| Boundary | Standalone behavior | Modular import behavior |
| --- | --- | --- |
| Numeric core | `cargo test --workspace`; maturin build | Same artifacts; no network at fit time |
| Python package | `pip install -e .` + `pytest` | Import `fast_mlsirm` without sibling repos |
| Governance workflows | Repo-local `.github/workflows` | Org reusable workflows from ContextualWisdomLab/.github when present |
| LLM automation | Optional; `NVIDIA_NIM_API_KEY` only | Prefer `contextual-orchestrator` when orchestration is required; do not use `COPILOT_GITHUB_TOKEN` for agent paths |
| Data / PII | Local process; no silent masking that blocks scoring | Access control + audit over irreversible PII redaction for production scoring paths |

## 4. Estimation & population structures

Supported population structures on the MMLE path (see `docs/mmle_marginal_lsirm_design.md`):

- **single** — independent persons
- **multigroup** — known group membership (DIF / equating contexts)
- **multilevel** — cluster random intercept \(u_c\) (school / class nesting)

Buyer gap (in flight): **multiple membership** and **longitudinal / temporal
occasion** *contracts* under `fast_mlsirm.multilevel` (content-addressed,
fail-closed) so atomistic fallacy is not forced by a single-level API. Nested
estimation that consumes those contracts remains paper-scoped in Rust.

## 5. Data flow (fit)

```text
responses Y [P×I] ──► validate (Python)
                         │
                         ▼
              FitConfig(estimator, backend, device, …)
                         │
         ┌───────────────┴────────────────┐
         ▼                                ▼
   backend=rust (default)           backend=numpy (parity)
   fast_mlsirm._core                pure-Python objective
         │                                │
         └──────────── fit result ────────┘
                         │
                         ▼
              FitResult + diagnostics + optional HTML report
```

Recovery evidence path used in CI and release acceptance:

```text
simulate(true θ, a, b, ξ, ζ) → fit/estimate → Procrustes align → RMSE / recovery metrics
```

## 6. Security & compliance posture

- **SAST / supply chain:** CodeQL, Semgrep, OSV, Trivy, Scorecard, Strix (org).
- **CSAP / SOC 2 awareness:** change control via PR + required checks; secrets
  never in tree; agent automation uses dedicated NVIDIA NIM credentials, not
  review-bot token schemes.
- **PII:** production scoring must not depend on irreversible masking that
  destroys person-level measurement; prefer encryption-at-rest / access
  control / purpose limitation documented in operability notes.
- **Input hardening:** bounded JSON / hostile control rejection on public
  parsers (see security tests).

## 7. Testing strategy (architecture-level)

| Layer | What must be true |
| --- | --- |
| Rust unit | Equation contracts, gradients, multilevel/MMLE edges |
| Recovery | True-parameter recovery / RMSE sentinels (not hard-coded theater) |
| Python API | Config fail-closed, public fit path, report accessibility |
| GPU | Explicit parity smoke vs CPU (Lavapipe in CI) |
| Fuzz | CSV / report / config Atheris budgets on every PR |
| CI matrix | CPython 3.12 **and** 3.14 full pytest; required check name `python` |

## 8. Repository map

```text
crates/mlsirm-core/     Rust formulas, GPU marginal, recovery tests
crates/fast-mlsirm-py/  PyO3 bindings
python/fast_mlsirm/     public API + orchestration
tests/                  Python contracts + recovery integration
docs/                   PRD/TRD, designs, doctoring (APA 7th)
docs/doctoring/         paper/standard citations for shipped behaviors
scripts/                release acceptance, sales readiness, changelog render
.github/workflows/      CI, security, governance agents
```

## 9. Governance documents index

| Artifact | Role |
| --- | --- |
| `AGENTS.md` / `CLAUDE.md` | Agent/developer operating rules |
| `ARCHITECTURE.md` (this file) | System structure |
| `CHANGELOG.md` + `docs/changelog.d/` | Fragment-sourced release notes |
| `docs/prd_trd_summary.md` | Product / technical requirements summary |
| `docs/doctoring/*` | APA 7th citations for model/security claims |
| `docs/20b_product_readiness.md` | Commercial readiness gate narrative |
| Threat model / test strategy | Evolved under `docs/` design notes + CI workflows |

## 10. References (APA 7th)

Fox, J.-P., & Glas, C. A. W. (2001). Bayesian estimation of a multilevel IRT
model. *Psychometrika, 66*(2), 271–288. https://doi.org/10.1007/BF02294839

Jeon, M., Jin, I. H., Schweinberger, M., & Baugh, S. (2021). Mapping unobserved
item-respondent interactions: A latent space item response model with
interaction map. *Psychometrika, 86*(2), 378–403.
https://doi.org/10.1007/s11336-021-09762-5

Kang, I., & Jeon, M. (2025). Multidimensional latent space item response models:
A note on the relativity of conditional dependence. *Psychometrika, 90*(2),
799–826. https://doi.org/10.1017/psy.2025.5

Molenaar, D., & Jeon, M. (2026). Regularized joint maximum likelihood estimation
of latent space item response models. *Psychometrika, 91*, 335–359.
https://doi.org/10.1017/psy.2025.10068
