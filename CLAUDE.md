# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Read AGENTS.md First

`AGENTS.md` at the repo root is the canonical agent operating guide. Read it
fully before making changes. In particular it defines:

- **Paper-first research**: changes to model formulas, estimators, fit
  diagnostics, simulation contracts, or interpretation-facing outputs must be
  grounded in the MLSIRM/MLS2PLM psychometric literature listed there.
- **Formula scope**: the implemented formula is a valid *simple-structure
  specialization* of MLS2PLM. Do not "fix" or reinterpret it through local
  gradient/distance/masking/vectorization edits; full MLS2PLM support must be a
  separate, complete model-design PR (parameterization, likelihood, analytic
  gradients, tests, docs, and Rust parity together).
- **Security gate**: every PR runs a central Security Scan (osv-scan,
  dependency-review, trivy-fs). A failing `trivy-fs` is a real finding —
  remediate by bumping the crate (`cargo update -p <crate>`) or Python
  dependency; never weaken the gate.
- **Ecosystem role**: fast-mlsirm calibrates LLM-as-a-Judge outputs and manages
  evaluation-item quality (aFIPC fixed-item calibration + kaefa item-fit) within
  the ContextualWisdomLab ecosystem.

## Common Commands

### Setup

An editable install builds the Rust extension with maturin, so a working Rust
toolchain should be on `PATH` (`cargo`/`rustc`) for deterministic local builds.
If cargo is absent, maturin may try to provision a temporary Rust toolchain via
`puccinialin`; set `MATURIN_NO_INSTALL_RUST=1` when you need a fail-fast
offline/proxy-safe build. A proxy or certificate error in that fallback is not
proof of a Python/PyO3 incompatibility. `pyproject.toml` declares Python
`>=3.10`; required CI currently builds and tests CPython 3.12, so broader
interpreter claims need matching build/import/full-suite CI evidence.

```bash
python -m pip install -e .          # builds fast_mlsirm._core via maturin
python -m pip install -e .[dev]     # adds pytest + hypothesis
```

### Tests

```bash
pytest                                        # Python suite (tests/)
pytest tests/test_objective.py                # single file
pytest tests/test_objective.py::test_name     # single test

cargo test --workspace                        # mlsirm-core (incl. proptest harness)
cargo test --manifest-path crates/fast-mlsirm-py/Cargo.toml   # PyO3 crate (workspace-excluded)
```

Note: the root `Cargo.toml` workspace contains only `crates/mlsirm-core` and
**excludes** `crates/fast-mlsirm-py`, so `cargo test --workspace` does not cover
the binding crate — CI tests it with the explicit `--manifest-path` invocation.

`tests/test_rust_parity.py` is the Rust<->NumPy numerical parity gate (agreement
to 1e-6 across all five model variants); CI additionally asserts the Rust core
is the resolved default backend:

```bash
python -c "from fast_mlsirm.backend import resolve_backend; from fast_mlsirm import FitConfig; assert resolve_backend(FitConfig().backend) == 'rust'"
```

### CLI and examples

```bash
fast-mlsirm simulate --persons 500 --dims 2 --items-per-dim 8 --out runs/sim_001
fast-mlsirm fit --responses runs/sim_001/responses.npy --factors runs/sim_001/item_factor.csv --model MLS2PLM --backend auto --out runs/fit_001
python examples/rust_backend_smoke.py
```

All CLI commands (`simulate`, `fit`, `diagnose-fit`, `diagnose-dimensions`,
`diagnose-response-process`, `diagnose-response-candidates`,
`diagnose-fixed-item-calibration`, `render-report`) accept `--json` for
machine-readable output. See README.md for the full option sets.

### Fuzzing

```bash
python -m pip install -e .[fuzz]    # atheris + hypothesis
python fuzz/atheris/fuzz_config.py -max_total_time=60 -timeout=25 fuzz/corpus/config
python fuzz/atheris/fuzz_load_factor_csv.py -max_total_time=60 -timeout=25 fuzz/corpus/load_factor_csv
python fuzz/atheris/fuzz_render_report.py -max_total_time=60 -timeout=25 fuzz/corpus/render_report
```

The Rust libFuzzer target (`fuzz/fuzz_targets/neg_loglik.rs`) runs via
ClusterFuzzLite on PRs touching `crates/` or `fuzz/`. Fuzz contract (see
`fuzz/README.md`): on arbitrary input the code either succeeds or raises a
documented, benign exception; panics, hangs, and `AssertionError`/`KeyError`/
`IndexError`/`TypeError` are bugs.

### Release evidence (package CI job)

```bash
python scripts/release_acceptance.py --out acceptance_check --require-rust
python scripts/build_commercial_release.py --out commercial-release --require-rust --check-import
```

CI installs dependencies with `pip --require-hashes -r requirements/ci.txt`;
the hash-locked `requirements/*.txt` files are generated from the
`requirements/*.in` inputs — update both when changing pins.

## Architecture

fast-mlsirm is a toolkit for **Multidimensional Latent Space Item Response
Models** (MLS2PLM and related constraints: `MIRT`, `MLSRM`, `MLS2PLM`, `ULSRM`,
`ULS2PLM`): binary response simulation, regularized JML/MAP-style point
estimation (Adam + small L-BFGS, no SciPy), recovery/fit/dimensionality
diagnostics, aFIPC-style fixed-item calibration diagnostics, and standalone HTML
reports. It is intentionally *not* a Bayesian sampler.

### Layout and how the pieces connect

```text
python/fast_mlsirm/       Python public API, CLI, and NumPy reference backend
crates/mlsirm-core/       Rust likelihood/gradient core (+ optional wgpu GPU path)
crates/fast-mlsirm-py/    PyO3 cdylib binding, built by maturin as fast_mlsirm._core
tests/                    Python suite, incl. the Rust<->NumPy parity gate
fuzz/                     Atheris harnesses, cargo-fuzz target, corpora
scripts/                  Release-acceptance / sales-readiness evidence builders
docs/                     PRD/TRD summary, design docs, papers, readiness gates
examples/enterprise_demo/ Synthetic procurement evidence manifests
```

- The **Rust core is the primary numeric path**. `mlsirm-core` implements the
  negative log-likelihood, analytic gradients, and distance kernels;
  `fast-mlsirm-py` exposes them to Python (`neg_loglik_and_grad`) via
  PyO3/numpy. maturin (configured in `pyproject.toml`) compiles the extension
  into `fast_mlsirm._core` during install.
- The **backend axis is `{numpy, rust, auto}`** (default `auto`), resolved in
  `python/fast_mlsirm/backend.py`: Rust when `_core` imports, otherwise the
  numerically-identical NumPy reference in `python/fast_mlsirm/objective.py`
  and `math.py`, which is kept for parity testing and fallback.
- **GPU is a device sub-option of the Rust backend**, not a separate backend:
  `FitConfig(backend="rust", rust_device={auto,cpu,gpu})`. The wgpu kernels in
  `crates/mlsirm-core/src/gpu.rs` run in f32 and fall back to the f64 scalar
  CPU reference at runtime when no adapter is present (so CI passes unchanged).
- **Entry points**: the `fast-mlsirm` console script maps to
  `fast_mlsirm.cli:main`; the Python API surface is re-exported from
  `python/fast_mlsirm/__init__.py` (`simulate`, `fit`, `fit_diagnostics`,
  `dimensionality_diagnostics`, `recovery_report`,
  `fixed_item_calibration_diagnostics`, `render_diagnostics_report`, ...).

### Key conventions

- **Rust/NumPy parity is a hard invariant.** Any change to the objective,
  gradients, or distance kernels must be mirrored across both backends and keep
  `tests/test_rust_parity.py` green. Formula-contract changes fall under the
  AGENTS.md model-design PR rule.
- **Missing responses** are excluded via `NaN`, `-1`, or an explicit mask; the
  mask semantics are part of the numeric contract shared by both backends.
- `.jules/` holds accumulated agent learnings worth honoring: `bolt.md`
  (NumPy performance patterns — avoid intermediate allocations, prefer
  einsum/BLAS-backed forms), `sentinel.md` (security — `np.load(...,
  allow_pickle=False)`, no `assert` for runtime checks, validate URI schemes,
  bound user-derived array dimensions), `palette.md` (HTML report
  accessibility).
- CI gates (`.github/workflows/ci.yml`): `python` (editable install +
  rust-default assertion + pytest), `rust` (both cargo test invocations),
  `fuzz` (bounded Atheris runs), `package` (wheel build + release acceptance +
  sales-readiness manifest). Plus ClusterFuzzLite PR fuzzing and the central
  Security Scan described in AGENTS.md.
