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


def _core_module():
    try:
        from . import _core  # type: ignore

        return _core
    except Exception:  # pragma: no cover
        return None


def serving_prior(bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    """Default scoring prior implied by the bundle's population block:
    N(0, 1) for single/multigroup-reference; the MARGINAL
    N(0, sqrt(1 + sigma_u^2)) for multilevel (unknown cluster). Pass an
    explicit prior to ``score_respondents`` to condition on a known cluster
    (mean = u_eap) or group (mean = mu_g, sd = sigma_g).
    """
    n_dims = bundle["n_dims"]
    mean = np.zeros(n_dims)
    sd = np.ones(n_dims)
    pop = bundle.get("population") or {}
    if pop.get("kind") == "multilevel" and "sigma_u" in pop:
        sd[:] = float(np.sqrt(1.0 + pop["sigma_u"] ** 2))
    return mean, sd


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
        "eapsum_tables": None,
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
        if "pi_zero" in pop:
            # zero-inflated calibration: serving scores are conditional on the
            # engager class; pi is reported for downstream base-rate handling
            out_pop["pi_zero"] = float(pop["pi_zero"])
        if "delta" in pop:
            out_pop["covariate_delta"] = float(pop["delta"])
        bundle["population"] = out_pop
    # Summed-score EAP conversion tables (Lord-Wingersky / Thissen et al.
    # 1995) under the bundle's serving prior — the lookup-table serving path.
    core = _core_module()
    if core is not None:
        mean, sd = serving_prior(bundle)
        zeta = np.asarray(p.zeta, dtype=np.float64)
        tables = core.eapsum_tables(
            np.asarray(p.alpha, dtype=np.float64),
            np.asarray(p.b, dtype=np.float64),
            zeta.ravel(),
            float(p.tau),
            factor_id,
            result.model,
            int(factor_id.max()) + 1,
            int(zeta.shape[1]),
            float(eps_distance),
            mean,
            sd,
            q_theta=int(q_theta),
            q_xi=int(q_xi),
        )
        bundle["eapsum_tables"] = [
            {
                "dim": int(t["dim"]),
                "dim_name": None if dim_names is None else dim_names[int(t["dim"])],
                "n_items_dim": int(t["n_items_dim"]),
                "score_prob": [float(v) for v in t["score_prob"]],
                "eap": [float(v) for v in t["eap"]],
                "sd": [float(v) for v in t["sd"]],
            }
            for t in tables
        ]
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
    method: str = "eap",
    prior: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """Score new respondents against a frozen bundle.

    ``responses`` is either a dense array (persons x n_items, NaN = missing,
    column order = bundle item order) or one/many dicts mapping item code ->
    0/1 (missing items simply absent) — the same shape of payload the
    importance-assessment API receives.

    ``method`` is "eap" (posterior mean, default), "map" (posterior mode with
    SEs), or "eapsum" (summed-score lookup via the bundle's Lord-Wingersky
    conversion tables — requires complete responses within each dimension).
    ``prior`` overrides the serving prior (mean, sd per dimension): condition
    on a known team with ``mean = u_eap`` or a known group with
    ``(mu_g, sigma_g)``.
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
    n_dims = bundle["n_dims"]
    mean, sd = serving_prior(bundle) if prior is None else (
        np.asarray(prior[0], dtype=float),
        np.asarray(prior[1], dtype=float),
    )

    if method == "eapsum":
        tables = bundle.get("eapsum_tables")
        if not tables:
            raise ValueError("bundle has no eapsum_tables; re-export the bundle")
        results = []
        for r in range(y.shape[0]):
            theta, theta_sd = [], []
            for t in sorted(tables, key=lambda t: t["dim"]):
                d_items = [j for j, it in enumerate(items) if it["factor_id"] == t["dim"]]
                if not all(observed[r, j] for j in d_items):
                    raise ValueError(
                        "eapsum scoring requires complete responses within each dimension"
                    )
                score = int(sum(y[r, j] for j in d_items))
                theta.append(float(t["eap"][score]))
                theta_sd.append(float(t["sd"][score]))
            results.append(
                {
                    "theta": theta,
                    "theta_sd": theta_sd,
                    "method": "eapsum",
                    "n_observed": int(observed[r].sum()),
                }
            )
        return results

    core = _core_module()
    n_persons = y.shape[0]
    y_filled = np.where(observed, y, 0.0)
    if method == "map":
        if core is None:
            raise ValueError("MAP scoring requires the compiled Rust core")
        res = core.score_bank_map(
            y_filled.ravel(), observed.ravel(), int(n_persons),
            alpha, b, zeta.ravel(), float(bundle["tau"]), factor_id,
            bundle["model"], int(n_dims), int(bundle["latent_dim"]),
            float(bundle["eps_distance"]), mean, sd,
        )
        theta_map = np.asarray(res["theta_map"]).reshape(n_persons, n_dims)
        theta_se = np.asarray(res["theta_se"]).reshape(n_persons, n_dims)
        xi_map = np.asarray(res["xi_map"]).reshape(n_persons, bundle["latent_dim"])
        return [
            {
                "theta": [float(v) for v in theta_map[r]],
                "theta_sd": [float(v) for v in theta_se[r]],
                "xi": [float(v) for v in xi_map[r]],
                "log_posterior": float(res["log_posterior"][r]),
                "converged": bool(res["converged"][r]),
                "method": "map",
                "n_observed": int(observed[r].sum()),
            }
            for r in range(n_persons)
        ]
    if method != "eap":
        raise ValueError("method must be one of ['eap', 'map', 'eapsum']")

    if core is not None:
        res = core.score_bank_eap(
            y_filled.ravel(), observed.ravel(), int(n_persons),
            alpha, b, zeta.ravel(), float(bundle["tau"]), factor_id,
            bundle["model"], int(n_dims), int(bundle["latent_dim"]),
            float(bundle["eps_distance"]), mean, sd,
            q_theta=int(bundle["quadrature"]["q_theta"]),
            xi_rule="gh",
            q_xi=int(bundle["quadrature"]["q_xi"]),
        )
        out = {
            "theta_eap": np.asarray(res["theta_eap"]).reshape(n_persons, n_dims),
            "theta_sd": np.asarray(res["theta_sd"]).reshape(n_persons, n_dims),
            "xi_eap": np.asarray(res["xi_eap"]).reshape(
                n_persons, bundle["latent_dim"]
            ),
            "loglik": np.asarray(res["loglik"]),
        }
    else:
        if not (np.allclose(mean, 0.0) and np.allclose(sd, 1.0)):
            raise ValueError(
                "non-standard scoring priors require the compiled Rust core"
            )
        out = score_eap(
            y_filled,
            observed,
            factor_id,
            alpha,
            b,
            zeta,
            bundle["tau"],
            model=bundle["model"],
            n_dims=n_dims,
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
                "method": "eap",
                "n_observed": int(observed[r].sum()),
            }
        )
    return results


def _bundle_bank_args(bundle: dict[str, Any]) -> dict[str, Any]:
    items = bundle["items"]
    return dict(
        alpha=np.array([it["alpha"] for it in items], dtype=np.float64),
        b=np.array([it["b"] for it in items], dtype=np.float64),
        zeta=np.array([it["zeta"] for it in items], dtype=np.float64).ravel(),
        tau=float(bundle["tau"]),
        factor_id=np.array([it["factor_id"] for it in items], dtype=np.int64),
        model=bundle["model"],
        n_dims=int(bundle["n_dims"]),
        latent_dim=int(bundle["latent_dim"]),
        eps_distance=float(bundle["eps_distance"]),
    )


def bank_information(
    bundle: dict[str, Any], theta: np.ndarray, xi: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Item/test information at the given trait points (Magis 2013 formula;
    Lord's test-information tradition). ``theta`` is points x n_dims; ``xi``
    defaults to the origin of the latent space."""
    core = _core_module()
    if core is None:
        raise RuntimeError("bank_information requires the compiled Rust core")
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim == 1:
        theta = theta[:, None]
    n_points = theta.shape[0]
    if xi is None:
        xi = np.zeros((n_points, bundle["latent_dim"]))
    res = dict(
        core.bank_information(
            theta.ravel(), np.asarray(xi, dtype=np.float64).ravel(), int(n_points),
            **_bundle_bank_args(bundle),
        )
    )
    return {
        "item_info": np.asarray(res["item_info"]).reshape(n_points, bundle["n_items"]),
        "test_info": np.asarray(res["test_info"]).reshape(n_points, bundle["n_dims"]),
    }


def cat_next_item(
    bundle: dict[str, Any],
    responses_so_far: dict[str, Any],
    prior: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Adaptive-EAP CAT step over the frozen bank (Bock & Mislevy 1982;
    multidimensional targeting per Wang, Kuo & Chao 2010): returns the EAP
    state, the targeted dimension, and unadministered items ranked by
    information. ``responses_so_far`` maps item code -> 0/1."""
    core = _core_module()
    if core is None:
        raise RuntimeError("cat_next_item requires the compiled Rust core")
    items = bundle["items"]
    n_items = bundle["n_items"]
    code_to_col = {it["code"]: j for j, it in enumerate(items)}
    y = np.zeros(n_items)
    administered = np.zeros(n_items, dtype=bool)
    for code, value in responses_so_far.items():
        j = code_to_col.get(code)
        if j is None:
            raise ValueError(f"unknown item code {code!r}")
        y[j] = float(bool(value)) if isinstance(value, bool) else float(value)
        administered[j] = True
    mean, sd = serving_prior(bundle) if prior is None else (
        np.asarray(prior[0], dtype=float), np.asarray(prior[1], dtype=float))
    res = dict(
        core.cat_next_item(
            y, administered, prior_mean=mean, prior_sd=sd,
            q_theta=int(bundle["quadrature"]["q_theta"]), xi_rule="gh",
            q_xi=int(bundle["quadrature"]["q_xi"]),
            **_bundle_bank_args(bundle),
        )
    )
    res["ranked_codes"] = [items[i]["code"] for i in res["ranked_items"]]
    return res


def plausible_values(
    bundle: dict[str, Any],
    responses: dict[str, Any] | list[dict[str, Any]] | np.ndarray,
    n_draws: int = 5,
    seed: int = 1,
    prior: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Posterior plausible-value draws (Marsman et al. 2016) for secondary
    analyses; returns persons x n_draws x n_dims."""
    core = _core_module()
    if core is None:
        raise RuntimeError("plausible_values requires the compiled Rust core")
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
    observed = ~np.isnan(y)
    mean, sd = serving_prior(bundle) if prior is None else (
        np.asarray(prior[0], dtype=float), np.asarray(prior[1], dtype=float))
    pv = core.plausible_values(
        np.where(observed, y, 0.0).ravel(), observed.ravel(), int(y.shape[0]),
        prior_mean=mean, prior_sd=sd,
        q_theta=int(bundle["quadrature"]["q_theta"]), xi_rule="gh",
        q_xi=int(bundle["quadrature"]["q_xi"]), n_draws=int(n_draws), seed=int(seed),
        **_bundle_bank_args(bundle),
    )
    return np.asarray(pv).reshape(y.shape[0], n_draws, bundle["n_dims"])
