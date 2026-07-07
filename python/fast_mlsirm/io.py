from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .types import FitResult, MLSIRMParams, SimulationData


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


def load_params(path: str | Path) -> MLSIRMParams:
    # Security: explicitly disable pickle to prevent arbitrary code execution
    data = np.load(path, allow_pickle=False)
    return MLSIRMParams(theta=data["theta"], alpha=data["alpha"], b=data["b"], xi=data["xi"], zeta=data["zeta"], tau=float(data["tau"]))


def load_factor_csv(path: str | Path) -> np.ndarray:
    rows = Path(path).read_text(encoding="utf-8").strip().splitlines()
    if not rows:
        raise ValueError("factor CSV is empty")
    return np.array([int(line.split(",")[1]) for line in rows[1:]], dtype=np.int64)


def _write_factor_csv(path: Path, factor_id: np.ndarray) -> None:
    lines = ["item_id,factor_id"]
    lines.extend(f"{idx},{int(factor)}" for idx, factor in enumerate(factor_id))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
