from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .types import DimensionalityDiagnostics, FitDiagnostics, FitResult, MLSIRMParams, SimulationData


def save_simulation(data: SimulationData, run_dir: str | Path) -> None:
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(data.config), indent=2), encoding="utf-8")
    np.save(out / "responses.npy", data.Y)
    np.savez(
        out / "truth.npz",
        theta=data.truth.theta,
        alpha=data.truth.alpha,
        a=data.truth.a,
        b=data.truth.b,
        xi=data.truth.xi,
        zeta=data.truth.zeta,
        tau=np.array(data.truth.tau),
        gamma=np.array(data.truth.gamma),
        factor_id=data.factor_id,
        Phi=data.Phi,
    )
    _write_factor_csv(out / "item_factor.csv", data.factor_id)
    manifest = {
        "package": "fast-mlsirm",
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "MLS2PLM",
        "n_persons": int(data.Y.shape[0]),
        "n_items": int(data.Y.shape[1]),
        "n_dims": int(data.config.n_dims),
        "latent_dim": int(data.config.latent_dim),
        "gamma": float(data.config.gamma),
        "phi": float(data.config.phi),
        "seed": int(data.config.seed),
        "files": {"responses": "responses.npy", "truth": "truth.npz", "factors": "item_factor.csv"},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def save_fit_result(result: FitResult, run_dir: str | Path) -> None:
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = result.params
    np.savez(out / "params.npz", theta=p.theta, alpha=p.alpha, a=p.a, b=p.b, xi=p.xi, zeta=p.zeta, tau=p.tau, gamma=p.gamma)
    summary = {
        "model": result.model,
        "optimizer": result.optimizer,
        "objective": result.objective,
        "convergence_status": result.convergence_status,
        "n_iter": result.n_iter,
        "final_loglik": result.loglik_trace[-1] if result.loglik_trace else None,
    }
    (out / "fit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_fit_diagnostics(diagnostics: FitDiagnostics, run_dir: str | Path) -> None:
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "itemfit": _arrays_to_lists(diagnostics.itemfit),
        "personfit": _arrays_to_lists(diagnostics.personfit),
        "factorfit": _arrays_to_lists(diagnostics.factorfit or {}),
        "categoryfit": _arrays_to_lists(diagnostics.categoryfit or {}),
        "groupfit": _arrays_to_lists(diagnostics.groupfit or {}),
        "clusterfit": _arrays_to_lists(diagnostics.clusterfit or {}),
        "group_itemfit": _arrays_to_lists(diagnostics.group_itemfit or {}),
        "cluster_itemfit": _arrays_to_lists(diagnostics.cluster_itemfit or {}),
        "model_fit": diagnostics.model_fit,
    }
    (out / "fit_diagnostics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_dimensionality_diagnostics(diagnostics: DimensionalityDiagnostics, run_dir: str | Path) -> None:
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"candidates": diagnostics.candidates, "best": diagnostics.best}
    (out / "dimension_diagnostics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_params(path: str | Path) -> MLSIRMParams:
    # Security: explicitly disable pickle to prevent arbitrary code execution
    data = np.load(path, allow_pickle=False)
    return MLSIRMParams(theta=data["theta"], alpha=data["alpha"], b=data["b"], xi=data["xi"], zeta=data["zeta"], tau=float(data["tau"]))


def load_factor_csv(path: str | Path) -> np.ndarray:
    import warnings
    content = Path(path).read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError("factor CSV is empty")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.loadtxt(path, delimiter=',', skiprows=1, usecols=1, dtype=np.int64, ndmin=1)


def _write_factor_csv(path: Path, factor_id: np.ndarray) -> None:
    item_ids = np.arange(len(factor_id))
    data = np.column_stack((item_ids, factor_id))
    np.savetxt(
        path,
        data,
        delimiter=',',
        header='item_id,factor_id',
        comments='',
        fmt='%d'
    )


def _arrays_to_lists(values: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {key: np.asarray(value, dtype=float).tolist() for key, value in values.items()}
