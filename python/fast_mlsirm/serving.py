"""Serving bundle export and frozen-parameter scoring.

The downstream deployment pattern (mirroring the mirt-based R plumber API
this feeds): a calibration run freezes the item-side parameters into a single
self-contained JSON bundle; a scoring service loads the bundle and computes
EAP trait scores / latent-space positions for new response vectors, never
re-estimating item parameters.

Bundle schema (``schema_version`` 1): model/config block, ordered item codes,
item parameters (``alpha``/``a``/``b``/``zeta``), ``tau``/``gamma``,
population block (multigroup ``mu``/``sigma``, multilevel ``sigma_u``/
``icc``), quadrature spec, and an optional item-screening audit trail.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .estimators.marginal import score_eap
from .types import FitResult

SCHEMA_VERSION = 1


def export_serving_bundle(
    result: FitResult,
    item_codes: list[str],
    factor_id: np.ndarray,
    path: str | Path | None = None,
    q_theta: int = 21,
    q_xi: int = 11,
    eps_distance: float = 1e-8,
    screening_audit: dict[str, Any] | None = None,
    dim_names: list[str] | None = None,
) -> dict[str, Any]:
    """Build (and optionally write) the serving bundle for a marginal fit."""
    p = result.params
    n_items = len(p.b)
    if len(item_codes) != n_items:
        raise ValueError("item_codes length must match the fitted item count")
    factor_id = np.asarray(factor_id, dtype=np.int64)
    if factor_id.shape != (n_items,):
        raise ValueError("factor_id length must match the fitted item count")
    bundle: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model": result.model,
        "estimator": "mmle",
        "optimizer": result.optimizer,
        "n_items": n_items,
        "n_dims": int(factor_id.max()) + 1,
        "latent_dim": int(np.asarray(p.zeta).shape[1]),
        "dim_names": dim_names,
        "quadrature": {"q_theta": q_theta, "q_xi": q_xi},
        "eps_distance": eps_distance,
        "items": [
            {
                "code": item_codes[i],
                "factor_id": int(factor_id[i]),
                "alpha": float(p.alpha[i]),
                "a": float(np.exp(p.alpha[i])),
                "b": float(p.b[i]),
                "zeta": [float(v) for v in np.asarray(p.zeta)[i]],
            }
            for i in range(n_items)
        ],
        "tau": float(p.tau),
        "gamma": float(np.exp(p.tau)),
        "population": None,
        "fit": {
            "convergence_status": result.convergence_status,
            "n_iter": result.n_iter,
            "final_loglik": result.loglik_trace[-1] if result.loglik_trace else None,
        },
        "screening_audit": screening_audit,
    }
    if result.population is not None:
        pop = dict(result.population)
        out_pop: dict[str, Any] = {"kind": pop["kind"]}
        if "mu" in pop:
            out_pop["mu"] = np.asarray(pop["mu"]).tolist()
            out_pop["sigma"] = np.asarray(pop["sigma"]).tolist()
        if "sigma_u" in pop:
            out_pop["sigma_u"] = float(pop["sigma_u"])
            out_pop["icc"] = float(pop["icc"])
        bundle["population"] = out_pop
    if path is not None:
        Path(path).write_text(
            json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return bundle


def load_serving_bundle(path: str | Path) -> dict[str, Any]:
    bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported bundle schema_version {bundle.get('schema_version')!r}"
        )
    return bundle


def score_respondents(
    bundle: dict[str, Any],
    responses: dict[str, Any] | list[dict[str, Any]] | np.ndarray,
    mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Score new respondents against a frozen bundle.

    ``responses`` is either a dense array (persons x n_items, NaN = missing,
    column order = bundle item order) or one/many dicts mapping item code ->
    0/1 (missing items simply absent) — the same shape of payload the
    importance-assessment API receives.
    """
    items = bundle["items"]
    n_items = bundle["n_items"]
    code_to_col = {it["code"]: j for j, it in enumerate(items)}
    if isinstance(responses, dict):
        responses = [responses]
    if isinstance(responses, list):
        y = np.full((len(responses), n_items), np.nan)
        for r, resp in enumerate(responses):
            for code, value in resp.items():
                j = code_to_col.get(code)
                if j is None:
                    raise ValueError(f"unknown item code {code!r}")
                y[r, j] = float(bool(value)) if isinstance(value, bool) else float(value)
    else:
        y = np.asarray(responses, dtype=float)
        if y.ndim == 1:
            y = y[None, :]
        if y.shape[1] != n_items:
            raise ValueError("responses column count must match the bundle items")
    observed = ~np.isnan(y) if mask is None else np.asarray(mask, dtype=bool)
    obs_vals = y[observed]
    if obs_vals.size and not np.all((obs_vals == 0.0) | (obs_vals == 1.0)):
        raise ValueError("observed responses must be 0 or 1")

    alpha = np.array([it["alpha"] for it in items])
    b = np.array([it["b"] for it in items])
    zeta = np.array([it["zeta"] for it in items])
    factor_id = np.array([it["factor_id"] for it in items], dtype=np.int64)
    out = score_eap(
        np.where(observed, y, 0.0),
        observed,
        factor_id,
        alpha,
        b,
        zeta,
        bundle["tau"],
        model=bundle["model"],
        n_dims=bundle["n_dims"],
        q_theta=bundle["quadrature"]["q_theta"],
        q_xi=bundle["quadrature"]["q_xi"],
        eps_distance=bundle["eps_distance"],
    )
    results = []
    for r in range(y.shape[0]):
        results.append(
            {
                "theta": [float(v) for v in out["theta_eap"][r]],
                "theta_sd": [float(v) for v in out["theta_sd"][r]],
                "xi": [float(v) for v in out["xi_eap"][r]],
                "loglik": float(out["loglik"][r]),
                "n_observed": int(observed[r].sum()),
            }
        )
    return results
