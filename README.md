# fast-mlsirm

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ContextualWisdomLab/fast-mlsirm)
[![CodSpeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://app.codspeed.io/ContextualWisdomLab/fast-mlsirm?utm_source=badge)


`fast-mlsirm` is an early high-performance toolkit for Multidimensional
Latent Space Item Response Models, focused on MLS2PLM simulation, regularized
point estimation, and true-parameter recovery checks.

The implemented simple-structure MLSIRM/MLS2PLM path follows Jeon, Jin,
Schweinberger, and Baugh (2021), Kang and Jeon (2025), and Molenaar and Jeon
(2026). Adjacent shipped methods include Angoff delta-plot DIF
([`docs/delta_plot_dif.md`](https://github.com/ContextualWisdomLab/fast-mlsirm/blob/main/docs/delta_plot_dif.md)) and Bradley–Terry MM
ranking ([`docs/bradley_terry_mm.md`](https://github.com/ContextualWisdomLab/fast-mlsirm/blob/main/docs/bradley_terry_mm.md)). Primary
citations and decision records live in
[`docs/traceability/research-basis.md`](https://github.com/ContextualWisdomLab/fast-mlsirm/blob/main/docs/traceability/research-basis.md)
and [`docs/adr/README.md`](https://github.com/ContextualWisdomLab/fast-mlsirm/blob/main/docs/adr/README.md). Score interpretation and
fairness remain governed by AERA, APA, and NCME (2014).

The public API stays small:

```python
import numpy as np

from fast_mlsirm import MLS2PLMConfig, FitConfig, fixed_item_calibration_diagnostics, simulate, dimensionality_diagnostics, fit, fit_diagnostics, recovery_report, render_diagnostics_report, response_process_dimensionality_diagnostics, response_process_fit_diagnostics

data = simulate(MLS2PLMConfig(seed=20260101))
result = fit(
    responses=data.Y,
    factor_id=data.factor_id,
    config=FitConfig(model="MLS2PLM", optimizer="adam_lbfgs", max_iter=100, backend="auto"),
)
report = recovery_report(data.truth, result.params)
diagnostics = fit_diagnostics(data.Y, result.params, data.factor_id, model=result.model)
dimensions = dimensionality_diagnostics(
    data.Y,
    data.factor_id,
    latent_dims=[1, 2, 3],
    config=FitConfig(model="MLS2PLM", optimizer="adam", max_iter=10, n_restarts=1),
)

print(report.summary)
print(diagnostics.model_fit)
print(dimensions.best)

category_probs = np.stack([1.0 - data.probabilities, data.probabilities], axis=2)
process_fit = response_process_fit_diagnostics(
    data.Y,
    category_probs,
    item_type="dichotomous",
    response_process="cumulative",
    group_id=np.arange(data.Y.shape[0]) % 2,
)
print(process_fit.itemfit["outfit_mnsq"])

process_dimensions = response_process_dimensionality_diagnostics(
    data.Y,
    {"dim2": category_probs},
    item_type="dichotomous",
    response_process="cumulative",
)
print(process_dimensions.best)

fixed_item_calibration = fixed_item_calibration_diagnostics(
    data.Y,
    {"dim2": category_probs},
    fixed_items=np.arange(min(4, data.Y.shape[1])),
    item_type="dichotomous",
    response_process="cumulative",
)
print(fixed_item_calibration.best)
```

## Features

- Canonical MLS2PLM binary response simulation.
- `gamma=0` no-CD simulation.
- Regularized JML/MAP-style fitting for `MIRT`, `MLSRM`, `MLS2PLM`,
  `ULSRM`, `ULS2PLM`, and `BIFAC2PLM` constraints.
- Missing response exclusion via `NaN`, `-1`, or an explicit mask, including
  missing-by-design rows or items when at least one response is observed.
- Adam and small L-BFGS-style optimizers without SciPy.
- Procrustes alignment and distance-based recovery metrics.
- Point-estimate item, person, and model fit diagnostics for fitted models.
- K-fold held-out likelihood diagnostics for latent-space dimensionality.
- Shared dichotomous/polytomous response-process diagnostics from category
  probabilities.
- Multigroup and multilevel-context fit summaries from person-level group or
  cluster IDs.
- Response-process probability candidate comparisons for external dimensionality
  checks.
- True-parameter reproduction, observed-information Hessian, vcov, standard
  error, and second-order stability helpers.
- Fixed item parameter linking, CAT item-information selection, and greedy ATA
  form assembly with content min/max constraints.
- Fixed-item calibration diagnostics that select candidate probability tensors
  using fixed evaluation-item likelihood and an item-fit penalty.
- Rubric-centered schemas, deterministic bounded item-blueprint compilation,
  and canonical provider-neutral generation contracts. See
  [Rubric-Centered Item Generation](https://github.com/ContextualWisdomLab/fast-mlsirm/blob/main/docs/rubric_item_generation.md).
- Provider-neutral contextual-orchestrator LLM-as-a-Judge integration with
  strict structured parsing. A judge result becomes an IRT row only through
  `LLMJudgeResult.to_irt_row()` with at least two criteria, followed by
  `validate_irt_response_matrix()`. Polytomous scoring defaults to
  `category_method="binary_threshold"`, where each ordered boundary is a
  bounded Boolean call and malformed or non-monotone evidence fails closed;
  `"direct"` and `"cumulative_threshold"` are explicit alternatives.
  `ContextualOrchestratorJudge` accepts only an adapter that declares the
  contextual-orchestrator contract, so a raw provider transport fails at
  construction. Category-count and prompt-perturbation calibration are required
  for every method, and paired controls are available through
  `build_multiple_choice_calibration_cases()` and
  `evaluate_paired_calibration()`. Paired score deltas are diagnostic
  sensitivity evidence, not a causal law or a claim of judge debiasing.
  The response-matrix and calibration contracts are specified in
  [ADR 0005](https://github.com/ContextualWisdomLab/contextual-orchestrator/blob/1b7dbd2a46533f41072def1fb94283147134cab5/docs/planning/adrs/0005-irt-response-matrix-contract.md),
  [ADR 0006](https://github.com/ContextualWisdomLab/contextual-orchestrator/blob/1b7dbd2a46533f41072def1fb94283147134cab5/docs/planning/adrs/0006-polytomous-llm-judge-bias-calibration.md), and
  [ADR 0008](https://github.com/ContextualWisdomLab/contextual-orchestrator/blob/1b7dbd2a46533f41072def1fb94283147134cab5/docs/planning/adrs/0008-fast-judge-review-hardening.md).
- Standalone HTML reports for saved fit or dimensionality diagnostics.
- CLI commands for simulation, fitting, diagnostics, and report rendering.
- Rust-backed fitting objective (neg-loglik, gradients, and distance kernels)
  via PyO3/maturin as the primary numeric path, with a numerically-identical
  NumPy reference backend kept for parity testing. `auto` fails closed when
  the compiled Rust core is unavailable.

## Install

```bash
python -m pip install fast-mlsirm
```

Python 3.12 or newer. Published wheels ship the compiled Rust core, so nothing
else is needed to fit a model.

The default runtime backend is `"auto"`. It uses the compiled Rust core
(`fast_mlsirm._core`) and fails closed when that extension is unavailable;
automatic resolution never silently selects NumPy. The NumPy reference path is
reached explicitly through `fast_mlsirm.fit_reference()` or
`fast-mlsirm fit --reference`, not by passing `backend="numpy"` to `fit()`.

For local development from a checkout:

```bash
python -m pip install -e ".[dev]"
cargo test --workspace
```

Source and editable installs build the extension with maturin, so they need a
working Rust toolchain; the PyO3 crate is exercised by the Python parity
tests.

## Project Status

`fast-mlsirm` is an alpha research library. It is usable today by teams that
run local MLS2PLM simulation, point-estimate fitting, diagnostics, and report
generation, and it is not a regulated decision product, a hosted assessment
platform, or a Bayesian posterior inference engine.

Every tagged release is built from a wheel that carries the compiled Rust core,
and is checked against the Rust/NumPy parity suite plus a release-acceptance
run before publication. The procedure lives in the
[release acceptance guide](https://github.com/ContextualWisdomLab/fast-mlsirm/blob/main/docs/release_acceptance.md).

- [Security policy](https://github.com/ContextualWisdomLab/fast-mlsirm/blob/main/SECURITY.md)
- [Support policy](https://github.com/ContextualWisdomLab/fast-mlsirm/blob/main/SUPPORT.md)
- [Changelog](https://github.com/ContextualWisdomLab/fast-mlsirm/blob/main/CHANGELOG.md)
- [Design decisions](https://github.com/ContextualWisdomLab/fast-mlsirm/blob/main/docs/adr/README.md)

## CLI

```bash
fast-mlsirm simulate \
  --persons 500 \
  --dims 2 \
  --items-per-dim 8 \
  --latent-dim 2 \
  --phi 0.3 \
  --gamma 1.5 \
  --seed 20260101 \
  --out runs/sim_001

fast-mlsirm fit \
  --responses runs/sim_001/responses.npy \
  --factors runs/sim_001/item_factor.csv \
  --model MLS2PLM \
  --backend auto \
  --latent-dim 2 \
  --optimizer adam_lbfgs \
  --max-iter 100 \
  --out runs/fit_001

fast-mlsirm diagnose-fit \
  --responses runs/sim_001/responses.npy \
  --factors runs/sim_001/item_factor.csv \
  --params runs/fit_001/params.npz \
  --model MLS2PLM \
  --out runs/diagnostics_001

fast-mlsirm diagnose-dimensions \
  --responses runs/sim_001/responses.npy \
  --factors runs/sim_001/item_factor.csv \
  --latent-dims 1,2,3 \
  --folds 5 \
  --model MLS2PLM \
  --max-iter 100 \
  --out runs/dimensions_001

fast-mlsirm diagnose-response-process \
  --responses runs/sim_001/responses.npy \
  --probabilities runs/model_probabilities.npy \
  --item-type polytomous \
  --response-process cumulative \
  --group-id runs/group_id.npy \
  --cluster-id runs/school_id.npy \
  --out runs/process_fit_001

fast-mlsirm diagnose-response-candidates \
  --responses runs/sim_001/responses.npy \
  --candidate dim1=runs/prob_dim1.npy \
  --candidate dim2=runs/prob_dim2.npy \
  --item-type dichotomous \
  --response-process ideal_point \
  --out runs/process_dimensions_001

fast-mlsirm diagnose-fixed-item-calibration \
  --responses runs/sim_001/responses.npy \
  --candidate dim1=runs/prob_dim1.npy \
  --candidate dim2=runs/prob_dim2.npy \
  --fixed-items runs/fixed_items.npy \
  --item-type dichotomous \
  --response-process ideal_point \
  --itemfit-penalty-weight 1.0 \
  --out runs/fixed_item_calibration_001

fast-mlsirm render-report \
  --diagnostics runs/diagnostics_001/fit_diagnostics.json \
  --out runs/diagnostics_001/report.html
```

For automation, every CLI command also accepts `--json`. In JSON mode,
progress text is suppressed and stdout contains one status object with the
output directory, key metrics, and generated file paths:

```bash
fast-mlsirm simulate \
  --persons 500 \
  --dims 2 \
  --items-per-dim 8 \
  --out runs/sim_001 \
  --json

fast-mlsirm fit \
  --responses runs/sim_001/responses.npy \
  --factors runs/sim_001/item_factor.csv \
  --out runs/fit_001 \
  --json
```

`fit`, `diagnose-fit`, and `diagnose-dimensions` validate that `responses.npy`
is a 2D persons-by-items matrix and that `item_factor.csv` has exactly one
factor id per item before running optimization or diagnostics.
`diagnose-fixed-item-calibration` writes `dimension_diagnostics.json` with
`best_candidate`, `calibration_score`, fixed-item coverage counts, and
item-fit penalty metrics. `--fixed-items` accepts a `.npy` boolean
mask or item-index vector; when omitted, all items are treated as the fixed
calibration set.

The explicit NumPy reference objective runs through `fast_mlsirm.fit_reference`
(Python) or the CLI's `fast-mlsirm fit --reference` flag; production `--backend`
choices are `{rust, auto}`. `fit --backend rust`
requires the installed `fast_mlsirm._core` extension and fails clearly if it is
unavailable. `fit --backend auto` uses the Rust objective when the compiled
core is available and fails closed otherwise. Automatic resolution never
silently selects NumPy.

The production backend axis is `{rust, auto}`; NumPy stays reachable only
through the reference API above. GPU acceleration is a *device* sub-option of
the Rust backend rather than a separate backend, selected with
`fit --backend rust --rust-device {auto,cpu,gpu}` (or `FitConfig(backend="rust",
rust_device=...)`). The Rust core carries a [wgpu](https://github.com/gfx-rs/wgpu)
(MIT/Apache-2.0) GPGPU implementation of the negative-log-likelihood and gradient
hot path:

- `rust_device="auto"` (default) runs the GPGPU kernels when a compatible GPU
  adapter is present and otherwise falls back to the identical CPU path — no
  GPU required.
- `rust_device="gpu"` prefers the GPU and still falls back to CPU (with a
  warning) when none is available, so CI and GPU-less machines pass unchanged.
- `rust_device="cpu"` always uses the scalar CPU reference.

The GPU kernels run in single precision (WGSL has no `f64`); the CPU path is the
`f64` reference the numerical-parity tests assert against. The requested Rust
device is recorded on `FitResult.rust_device` and in `fit_summary.json`; when
`gpu` is explicitly requested on a machine without a compatible adapter, the
runtime prints a warning and falls back to the CPU implementation.

`render-report` turns `fit_diagnostics.json` or `dimension_diagnostics.json`
into a standalone HTML report with model summary cards, compact tables, and
small bar views when chartable diagnostic metrics are present. Optional fit
tables, dimensionality candidate comparisons, or metric summaries without
values are summarized in a diagnostics coverage block instead of rendering as
repeated blank-looking report sections or placeholder-only columns.

## Repository Layout

```text
python/fast_mlsirm/       Python public API and reference backend
crates/mlsirm-core/       Rust likelihood and gradient core
crates/fast-mlsirm-py/    PyO3 binding for the compiled Rust backend
tests/                    Python smoke and numerical tests
docs/                     Design docs, ADRs, and research traceability
```

## Scope

This is not a Bayesian sampler. The package provides fast simulation,
regularized point estimation, and recovery diagnostics, and the Rust backend
keeps the same point-estimate formula contract as the NumPy reference path.
Block-mode Rust execution, sparse response storage, posterior predictive
checking, and new ordinal response estimators are out of scope for the current
release.
