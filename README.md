# fast-mlsirm

`fast-mlsirm` is an early high-performance toolkit for Multidimensional
Latent Space Item Response Models, focused on MLS2PLM simulation, regularized
point estimation, and true-parameter recovery checks.

The first implementation keeps the public API small:

```python
import numpy as np

from fast_mlsirm import MLS2PLMConfig, FitConfig, simulate, dimensionality_diagnostics, fit, fit_diagnostics, recovery_report, response_process_dimensionality_diagnostics, response_process_fit_diagnostics

data = simulate(MLS2PLMConfig(seed=20260101))
result = fit(
    responses=data.Y,
    factor_id=data.factor_id,
    config=FitConfig(model="MLS2PLM", optimizer="adam_lbfgs", max_iter=100),
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
```

## What Works Now

- Canonical MLS2PLM binary response simulation.
- `gamma=0` no-CD simulation.
- Regularized JML/MAP-style fitting for `MIRT`, `MLSRM`, `MLS2PLM`,
  `ULSRM`, and `ULS2PLM` constraints.
- Missing response exclusion via `NaN`, `-1`, or an explicit mask.
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
- CLI commands for simulation and fitting.
- Rust core crate with the same likelihood and gradient formulas.

## Install

For local development:

```bash
python -m pip install -e .
```

The Python reference backend requires NumPy. The Rust crate can be tested with:

```bash
cargo test
```

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

## Repository Layout

```text
python/fast_mlsirm/       Python public API and reference backend
crates/mlsirm-core/       Rust likelihood and gradient core
tests/                    Python smoke and numerical tests
docs/                     PRD/TRD summary and roadmap
```

## MVP Boundary

This is not a Bayesian sampler. The package intentionally starts with fast
simulation, regularized point estimation, and recovery diagnostics. PyO3
zero-copy bindings, block-mode Rust execution, sparse response storage, and
benchmark automation are next-step work after the formulas and API stabilize.
