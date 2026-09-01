# fast-mlsirm

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ContextualWisdomLab/fast-mlsirm)

**Rust-first psychometric measurement, simulation, fitting, diagnostics, and recovery evidence behind a Python API.**

`fast-mlsirm` is a reusable measurement core for teams that need explicit model assumptions, reproducible estimation, true-parameter recovery checks, and evidence-bound diagnostics without moving production numerical work into a hosted application.

The package is centered on multidimensional latent-space item-response modeling, with additional bounded measurement, calibration, linking, diagnostic, and evaluation contracts. Performance- and result-critical numerical paths are owned by Rust/PyO3; Python provides the public package surface, validation, orchestration, and reference/parity support.

## Why it exists

A psychometric library is useful only when callers can tell **what was estimated, by which model and backend, from which evidence, under which assumptions, and with what validation**. `fast-mlsirm` treats those questions as product contracts rather than leaving them to notebook convention.

| Need | What `fast-mlsirm` provides |
| --- | --- |
| High-performance measurement | Rust-backed estimation and numerical kernels exposed through PyO3 |
| Reproducible research | Deterministic simulation, explicit configuration, and true-parameter recovery workflows |
| Model diagnostics | Fit, dimensionality, response-process, calibration, and stability evidence |
| Governed measurement inputs | Explicit response/evidence contracts with bounded, fail-closed validation |
| Integration | A standalone Python package that downstream products can consume through versioned artifacts |
| Scientific traceability | Model equations, research basis, ADRs, and executable validation kept alongside source |

## Product boundary

`fast-mlsirm` owns **domain-neutral psychometric computation and reusable measurement contracts**. It does not own the surrounding participant or assessment lifecycle.

```text
assessment / research evidence
            │
            ▼
┌───────────────────────────────┐
│          fast-mlsirm          │
│ reusable measurement core     │
├───────────────────────────────┤
│ input / evidence validation   │
│ simulation                    │
│ Rust numerical estimation     │
│ scoring / diagnostics         │
│ recovery / comparison         │
│ versioned result evidence     │
└───────────────┬───────────────┘
                │
       versioned handoff
                ▼
 hosted products / research workflows
```

A downstream product such as `psychometrics-commons` owns, when applicable, accounts and authorization, participants and sessions, consent and data rights, hosted persistence, restricted item content, operational workflows, billing, human decisions, and deployment-specific regulatory controls.

Temporal/event semantics belong to [`TEPP`](https://github.com/ContextualWisdomLab/TEPP); provider/model orchestration belongs to [`contextual-orchestrator`](https://github.com/ContextualWisdomLab/contextual-orchestrator). `fast-mlsirm` remains independently installable and does not require either service to execute its ordinary package-owned numerical work.

## Supported foundations

The implemented simple-structure MLSIRM/MLS2PLM line is grounded in the repository's cited research, including Jeon, Jin, Schweinberger, and Baugh (2021), Kang and Jeon (2025), and Molenaar and Jeon (2026). The canonical research map is [`docs/traceability/research-basis.md`](docs/traceability/research-basis.md); architecture decisions are indexed in [`docs/adr/README.md`](docs/adr/README.md).

Current source also includes bounded capabilities around areas such as:

- MLSIRM/MLS2PLM and related constrained item-response fitting;
- simulation and true-parameter recovery;
- item, person, model, and dimensionality diagnostics;
- response-process and fixed-item calibration diagnostics;
- linking and selected CAT/ATA utilities;
- explicit missing/nonresponse and measurement-state contracts;
- evidence-bound rater, rubric, and evaluation contracts;
- source-free HTML/evidence reports for review and release workflows.

Not every research candidate or open-PR model is part of a stable support promise. Protected source, immutable releases, the model-specific evidence record, and the versioned capability contracts remain authoritative.

## Quick start

The current source tree declares `fast-mlsirm` **0.9.1**, Python **3.12+**, and a Maturin/PyO3 build. Building from source therefore requires a working Rust toolchain.

```bash
python -m pip install -e .
```

A compact simulation → fit → recovery workflow:

```python
from fast_mlsirm import FitConfig, MLS2PLMConfig, fit, recovery_report, simulate

sample = simulate(MLS2PLMConfig(seed=20260101))

result = fit(
    responses=sample.Y,
    factor_id=sample.factor_id,
    config=FitConfig(
        model="MLS2PLM",
        optimizer="adam_lbfgs",
        max_iter=100,
        backend="auto",
    ),
)

recovery = recovery_report(sample.truth, result.params)
print(recovery.summary)
```

The default extension is the **PyO3 binding for the compiled Rust backend**. `backend="auto"` uses that compiled Rust core and **fails closed when that extension is unavailable**. Automatic resolution never silently selects NumPy; it **fails closed otherwise**. The NumPy backend is an explicit reference/parity path and must be requested explicitly.

## Common workflows

### Diagnose a fitted model

```python
from fast_mlsirm import fit_diagnostics

diagnostics = fit_diagnostics(
    sample.Y,
    result.params,
    sample.factor_id,
    model=result.model,
)

print(diagnostics.model_fit)
```

### Compare candidate dimensionalities

```python
from fast_mlsirm import dimensionality_diagnostics

comparison = dimensionality_diagnostics(
    sample.Y,
    sample.factor_id,
    latent_dims=[1, 2, 3],
    config=FitConfig(
        model="MLS2PLM",
        optimizer="adam",
        max_iter=10,
        n_restarts=1,
    ),
)

print(comparison.best)
```

For specialized workflows—DIF, Bradley–Terry ranking, response-process diagnostics, fixed-item calibration, item generation/evaluation, scoring, linking, CAT/ATA, sampling design, and commercial/release evidence—use the documentation map below rather than treating one README example as the full API contract.

## Numerical ownership and safety

Production result-affecting mathematical and psychometric computation is Rust-first. Python is used for package-facing validation, marshalling, orchestration, and explicit reference/parity paths.

The repository uses fail-closed contracts around input representation, missingness, resource bounds, result envelopes, provenance, and backend selection. These controls are intended to prevent malformed or stale evidence from silently becoming a plausible-looking estimate; they do not convert a statistical result into a validity, fairness, causal, or high-stakes decision claim.

## Scientific interpretation

A technically correct estimate is not automatically a validated score use.

`fast-mlsirm` separates:

- numerical correctness and parameter recovery;
- construct and response-process evidence;
- model fit and dimensionality evidence;
- transportability and fairness evidence;
- downstream decision utility and policy.

Score interpretation and fairness guidance follows the measurement standards cited by the repository, including AERA, APA, and NCME (2014). Security standards and controls are tracked separately; they are not substitutes for psychometric validity evidence.

## Current maturity

The package metadata declares **Development Status :: 3 - Alpha**. The repository contains a substantial reusable measurement core and extensive verification/release evidence machinery, but it does not claim universal technical GA, universal fairness/validity, a hosted assessment platform, or approval for regulated/high-stakes decisions.

A bounded capability can mature independently when its exact numerical owner, public contract, recovery/conformance evidence, compatibility matrix, security/provenance evidence, and release support are all established. See [`docs/product-technical-gap-baseline.md`](docs/product-technical-gap-baseline.md) for the current completion model and remaining evidence gaps.

## Verification

For source development, install the development dependencies and run the repository-owned checks rather than relying on an import smoke test alone:

```bash
python -m pip install -e '.[dev]'
python -m pytest
cargo test --workspace
cargo test --manifest-path crates/fast-mlsirm-py/Cargo.toml
```

The PyO3 crate is intentionally excluded from the root Cargo workspace, so its binding tests must be run through its manifest in addition to `cargo test --workspace`.

The protected CI surface adds package, Rust/PyO3, coverage, fuzz/security/static-analysis, artifact, and release-contract checks. Exact current workflow evidence is authoritative; results from a predecessor source head do not transfer after a change.

For a release-oriented local evidence build, the repository provides:

```bash
python scripts/build_commercial_release.py \
  --out commercial-release \
  --require-rust \
  --check-import
```

That command produces review evidence; it does not by itself prove deployment, customer adoption, regulatory approval, commercial transfer, or suitability for a specific high-stakes use.

## Commercial Readiness

**Enterprise Sales Readiness** is an evidence gate, not a sales, valuation, certification, or customer claim. The canonical gate is [`scripts/sales_readiness.py`](scripts/sales_readiness.py), driven by release evidence from [`scripts/release_acceptance.py`](scripts/release_acceptance.py). Higher-level procurement packets are assembled by [`scripts/build_release_evidence_index.py`](scripts/build_release_evidence_index.py), [`scripts/build_commercial_release.py`](scripts/build_commercial_release.py), [`scripts/build_procurement_due_diligence.py`](scripts/build_procurement_due_diligence.py), [`scripts/build_pr_queue_governance.py`](scripts/build_pr_queue_governance.py), and [`scripts/build_figma_evidence_sync.py`](scripts/build_figma_evidence_sync.py).

These tools keep technical evidence distinct from actual customer acceptance, deployment, transfer, revenue, or legal authority. See [`docs/commercial_readiness.md`](docs/commercial_readiness.md) and [`docs/enterprise_sales_readiness.md`](docs/enterprise_sales_readiness.md) for the bounded evidence contract.

## Documentation map

| Goal | Start here |
| --- | --- |
| Documentation home | [`docs/README.md`](docs/README.md) |
| Public documentation landing | [`docs/index.md`](docs/index.md) |
| Product requirements | [`docs/PRD.md`](docs/PRD.md) |
| Technical requirements | [`docs/TRD.md`](docs/TRD.md) |
| Architecture | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Architecture decisions | [`docs/adr/README.md`](docs/adr/README.md) |
| Research basis | [`docs/traceability/research-basis.md`](docs/traceability/research-basis.md) |
| MLS2PLM canonical equations | [`docs/papers/mls2plm-canonical-equations.md`](docs/papers/mls2plm-canonical-equations.md) |
| DIF | [`docs/delta_plot_dif.md`](docs/delta_plot_dif.md) |
| Bradley–Terry ranking | [`docs/bradley_terry_mm.md`](docs/bradley_terry_mm.md) |
| Rubric-centered generation | [`docs/rubric_item_generation.md`](docs/rubric_item_generation.md) |
| Current product/technical gaps | [`docs/product-technical-gap-baseline.md`](docs/product-technical-gap-baseline.md) |
| Release acceptance | [`docs/release_acceptance.md`](docs/release_acceptance.md) |
| Security | [`SECURITY.md`](SECURITY.md) |
| Support | [`SUPPORT.md`](SUPPORT.md) |
| Changelog | [`CHANGELOG.md`](CHANGELOG.md) |

## Contributing

Before changing a model, numerical path, public contract, or scientific claim, read the repository guidance, applicable PRD/TRD/ADR, canonical equations, and research traceability. Keep production numerical ownership in Rust, add or update recovery/conformance evidence with behavior changes, and update public documentation together with the contract it describes.

## License

`fast-mlsirm` is licensed under the [MIT License](LICENSE). Third-party dependencies retain their own license terms and must remain compatible with the repository's commercial-use and attribution policy.
