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
import math
from pathlib import Path
from typing import Any

import numpy as np

from .config import MAX_LATENT_DIM, VALID_MODELS
from .estimators.marginal import score_eap
from .io import _load_json_bounded
from .types import FitResult

SCHEMA_VERSION = 1
MAX_DRAWS = 100_000
MAX_INFORMATION_POINTS = 100_000
MAX_SCORE_CELLS = 20_000_000
MAX_SERVING_OUTPUT_CELLS = 20_000_000
MAX_ABS_LOG_SCALE = 100.0
MAX_ABS_ITEM_PARAMETER = 1_000_000.0


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
    if not isinstance(n_dims, int) or isinstance(n_dims, bool) or not (1 <= n_dims <= 64):
        raise ValueError("bundle n_dims must be an integer in 1..64")
    mean = np.zeros(n_dims)
    sd = np.ones(n_dims)
    pop = bundle.get("population")
    if pop is None:
        pop = {}
    elif not isinstance(pop, dict):
        raise ValueError("bundle population must be an object or null")
    if pop.get("kind") == "multilevel" and "sigma_u" in pop:
        su = pop["sigma_u"]
        # sigma_u is attacker-controlled in an untrusted bundle: a string
        # crashes ** with TypeError, 1e200 overflows, 1e150 poisons sd.
        if not _finite_number(su) or su < 0 or su > 1_000.0:
            raise ValueError("bundle population sigma_u must be a finite number in 0..1000")
        sd[:] = float(np.sqrt(1.0 + float(su) ** 2))
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
    """Build (and optionally write) a serving bundle from a converged marginal fit.

    Raises ``RuntimeError`` when calibration did not converge.  A frozen bundle
    is a deployment artifact, so unfinished parameters must not cross this API
    boundary merely because they are finite and serializable.
    """
    status = str(result.convergence_status).strip().lower()
    if status != "converged":
        trace = result.loglik_trace
        last_delta = (
            abs(float(trace[-1]) - float(trace[-2]))
            if len(trace) >= 2
            else float("nan")
        )
        raise RuntimeError(
            "export_serving_bundle requires converged calibration parameters; "
            f"status={status or 'unknown'}, n_iter={result.n_iter}, "
            f"last_loglik_delta={last_delta:.6g}"
        )
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


def _reject_nonfinite_json(literal: str) -> float:
    raise ValueError(f"serving bundle contains a non-finite JSON constant {literal!r}")


def _finite_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _validate_bundle(bundle: Any) -> None:
    """Validate a serving bundle's structure, sizes, and numeric domains
    before it is used to score untrusted respondents. Guards against oversized
    or inconsistent dimensions (multi-terabyte allocations / index errors) and
    unsafe item parameters (NaN/Inf scores) reaching the scoring core."""
    if not isinstance(bundle, dict):
        raise ValueError("serving bundle must be a JSON object")
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported bundle schema_version {bundle.get('schema_version')!r}"
        )
    population = bundle.get("population")
    if population is not None and not isinstance(population, dict):
        raise ValueError("bundle population must be an object or null")

    def _pos_int(key: str, hi: int) -> int:
        v = bundle.get(key)
        if not isinstance(v, int) or isinstance(v, bool) or not (1 <= v <= hi):
            raise ValueError(f"bundle {key} must be an integer in 1..{hi}")
        return v

    n_items = _pos_int("n_items", 100_000)
    n_dims = _pos_int("n_dims", 64)
    latent_dim = _pos_int("latent_dim", MAX_LATENT_DIM)
    if bundle.get("model") not in VALID_MODELS:
        raise ValueError(f"bundle model must be one of {sorted(VALID_MODELS)}")
    if (
        not _finite_number(bundle.get("tau"))
        or abs(float(bundle["tau"])) > MAX_ABS_LOG_SCALE
    ):
        raise ValueError(
            f"bundle tau must be in the safe numeric range "
            f"[-{MAX_ABS_LOG_SCALE}, {MAX_ABS_LOG_SCALE}]"
        )
    eps = bundle.get("eps_distance")
    if (
        not _finite_number(eps)
        or eps <= 0
        or float(eps) > MAX_ABS_ITEM_PARAMETER
    ):
        raise ValueError(
            f"bundle eps_distance must be in the safe numeric range "
            f"(0, {MAX_ABS_ITEM_PARAMETER}]"
        )
    quad = bundle.get("quadrature")
    if not isinstance(quad, dict):
        raise ValueError("bundle quadrature must be an object")
    for qk in ("q_theta", "q_xi"):
        if quad.get(qk) not in {7, 11, 15, 21, 31, 41}:
            raise ValueError(f"bundle quadrature {qk} must be one of 7,11,15,21,31,41")
    # Latent-space models score on a tensor Gauss-Hermite grid of
    # q_xi ** latent_dim points; reject combinations that would allocate an
    # astronomically large grid (e.g. 41**8 ~ 8e12).
    if bundle["model"] != "MIRT" and int(quad["q_xi"]) ** latent_dim > 1_000_000:
        raise ValueError("bundle q_xi ** latent_dim exceeds the serving grid limit")
    # The scoring core builds item-response tables of size
    # max(n_items, n_dims) * q_theta * n_xi; bound the product (55+ GB otherwise).
    n_xi = 1 if bundle["model"] == "MIRT" else int(quad["q_xi"]) ** latent_dim
    if max(n_items, n_dims) * int(quad["q_theta"]) * n_xi > 50_000_000:
        raise ValueError("bundle scoring-table size (items x q_theta x n_xi) exceeds the serving limit")
    items = bundle.get("items")
    if not isinstance(items, list) or len(items) != n_items:
        raise ValueError("bundle items must be a list of length n_items")
    seen: set = set()
    for j, it in enumerate(items):
        if not isinstance(it, dict):
            raise ValueError(f"bundle item {j} must be an object")
        code = it.get("code")
        if not isinstance(code, str) or code in seen:
            raise ValueError(f"bundle item {j} must have a unique string code")
        seen.add(code)
        fid = it.get("factor_id")
        if not isinstance(fid, int) or isinstance(fid, bool) or not (0 <= fid < n_dims):
            raise ValueError(f"bundle item {code!r} factor_id must be an int in 0..n_dims-1")
        for pk, bound in (
            ("alpha", MAX_ABS_LOG_SCALE),
            ("b", MAX_ABS_ITEM_PARAMETER),
        ):
            if (
                not _finite_number(it.get(pk))
                or abs(float(it[pk])) > bound
            ):
                raise ValueError(
                    f"bundle item {code!r} {pk} must be in the safe numeric "
                    f"range [-{bound}, {bound}]"
                )
        zeta = it.get("zeta")
        if (
            not isinstance(zeta, list)
            or len(zeta) != latent_dim
            or not all(
                _finite_number(z) and abs(float(z)) <= MAX_ABS_ITEM_PARAMETER
                for z in zeta
            )
        ):
            raise ValueError(
                f"bundle item {code!r} zeta must be {latent_dim} numbers in "
                f"the safe numeric range [-{MAX_ABS_ITEM_PARAMETER}, "
                f"{MAX_ABS_ITEM_PARAMETER}]"
            )


def load_serving_bundle(path: str | Path) -> dict[str, Any]:
    bundle = _load_json_bounded(
        path,
        source="serving bundle",
        parse_constant=_reject_nonfinite_json,
    )
    _validate_bundle(bundle)
    return bundle


def score_respondents(
    bundle: dict[str, Any],
    responses: dict[str, Any] | list[dict[str, Any]] | np.ndarray,
    mask: np.ndarray | None = None,
    method: str = "eap",
    prior: tuple[np.ndarray, np.ndarray] | None = None,
    device: str = "cpu",
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
    _validate_bundle(bundle)
    items = bundle["items"]
    n_items = bundle["n_items"]
    code_to_col = {it["code"]: j for j, it in enumerate(items)}
    if isinstance(responses, dict):
        responses = [responses]
    if isinstance(responses, list):
        # Bound the dense respondent matrix before allocating: len(responses)
        # and n_items are both request/bundle controlled (memory-exhaustion DoS).
        if len(responses) * n_items > MAX_SCORE_CELLS:
            raise ValueError(
                f"response matrix ({len(responses)} x {n_items}) exceeds the "
                f"{MAX_SCORE_CELLS}-cell scoring limit"
            )
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
        elif y.ndim != 2:
            raise ValueError(
                "responses must be a 1-D vector or 2-D persons x items matrix"
            )
        if y.shape[1] != n_items:
            raise ValueError("responses column count must match the bundle items")
        if y.size > MAX_SCORE_CELLS:
            raise ValueError(
                f"response matrix ({y.shape[0]} x {n_items}) exceeds the "
                f"{MAX_SCORE_CELLS}-cell scoring limit"
            )
    if mask is None:
        observed = ~np.isnan(y)
    else:
        observed = np.asarray(mask, dtype=bool)
        if observed.shape != y.shape:
            raise ValueError("mask shape must match responses")
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
            device=str(device),
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
    _validate_bundle(bundle)
    core = _core_module()
    if core is None:
        raise RuntimeError("bank_information requires the compiled Rust core")
    n_dims = int(bundle["n_dims"])
    latent_dim = int(bundle["latent_dim"])
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim == 1:
        if n_dims == 1:
            theta = theta[:, None]
        elif theta.shape == (n_dims,):
            theta = theta[None, :]
        else:
            raise ValueError("theta must have shape (points, n_dims)")
    if theta.ndim != 2 or theta.shape[1] != n_dims:
        raise ValueError("theta must have shape (points, n_dims)")
    n_points = theta.shape[0]
    if not (1 <= n_points <= MAX_INFORMATION_POINTS):
        raise ValueError(
            f"theta must contain between 1 and {MAX_INFORMATION_POINTS} points"
        )
    output_cells = n_points * (int(bundle["n_items"]) + n_dims)
    if output_cells > MAX_SERVING_OUTPUT_CELLS:
        raise ValueError(
            f"bank-information output size ({output_cells} cells) exceeds the "
            f"{MAX_SERVING_OUTPUT_CELLS}-cell serving limit"
        )
    if not np.all(np.isfinite(theta)):
        raise ValueError("theta must contain only finite values")
    if xi is None:
        xi_array = np.zeros((n_points, latent_dim))
    else:
        xi_array = np.asarray(xi, dtype=np.float64)
        if xi_array.ndim == 1:
            if latent_dim == 1 and xi_array.shape == (n_points,):
                xi_array = xi_array[:, None]
            elif n_points == 1 and xi_array.shape == (latent_dim,):
                xi_array = xi_array[None, :]
            else:
                raise ValueError("xi must have shape (points, latent_dim)")
        if xi_array.shape != (n_points, latent_dim):
            raise ValueError("xi must have shape (points, latent_dim)")
        if not np.all(np.isfinite(xi_array)):
            raise ValueError("xi must contain only finite values")
    res = dict(
        core.bank_information(
            theta.ravel(), xi_array.ravel(), int(n_points),
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
    """Run one adaptive-EAP CAT step over the frozen bank.

    Bock and Mislevy (1982) support the noniterative EAP score, while Wang et
    al. (2010) describe multidimensional CAT with information-based item
    selection. Selecting the dimension with the largest posterior SD is a
    repository policy, not a procedure prescribed by either source.
    ``responses_so_far`` maps item code to 0/1.

    References
    ----------
    Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability
    in a microcomputer environment. *Applied Psychological Measurement,
    6*(4), 431–444. https://doi.org/10.1177/014662168200600405

    Wang, H.-P., Kuo, B.-C., & Chao, R.-C. (2010). A multidimensional
    computerized adaptive testing system for enhancing the Chinese as second
    language proficiency test. In H. Fujita & J. Sasaki (Eds.), *Selected topics
    in education and educational technology* (pp. 245–252). WSEAS Press.
    """
    _validate_bundle(bundle)
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
        response = float(bool(value)) if isinstance(value, bool) else float(value)
        if not np.isfinite(response) or response not in (0.0, 1.0):
            raise ValueError("administered responses must be 0 or 1")
        y[j] = response
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
    """Draw posterior plausible values for secondary analyses.

    The fixed item bank and discrete quadrature-grid sampler are repository
    choices; this function does not propagate item-parameter uncertainty.
    Returns persons x n_draws x n_dims.

    References
    ----------
    Marsman, M., Maris, G., Bechger, T., & Glas, C. (2016). What can we learn
    from plausible values? *Psychometrika, 81*(2), 274–289.
    https://doi.org/10.1007/s11336-016-9497-x
    """
    core = _core_module()
    if core is None:
        raise RuntimeError("plausible_values requires the compiled Rust core")
    _validate_bundle(bundle)
    if not isinstance(n_draws, (int, np.integer)) or isinstance(
        n_draws, (bool, np.bool_)
    ):
        raise ValueError("n_draws must be an integer")
    draw_count = int(n_draws)
    if not (1 <= draw_count <= MAX_DRAWS):
        raise ValueError(f"n_draws must be between 1 and {MAX_DRAWS}")
    items = bundle["items"]
    n_items = bundle["n_items"]
    code_to_col = {it["code"]: j for j, it in enumerate(items)}
    if isinstance(responses, dict):
        responses = [responses]
    if isinstance(responses, list):
        # Bound the dense respondent matrix before allocating: len(responses)
        # and n_items are both request/bundle controlled (memory-exhaustion DoS).
        if len(responses) * n_items > MAX_SCORE_CELLS:
            raise ValueError(
                f"response matrix ({len(responses)} x {n_items}) exceeds the "
                f"{MAX_SCORE_CELLS}-cell scoring limit"
            )
        y = np.full((len(responses), n_items), np.nan)
        for r, resp in enumerate(responses):
            for code, value in resp.items():
                j = code_to_col.get(code)
                if j is None:
                    raise ValueError(f"unknown item code {code!r}")
                y[r, j] = float(bool(value)) if isinstance(value, bool) else float(value)
    else:
        y = np.asarray(responses, dtype=float)
        if y.ndim != 2 or y.shape[1] != n_items:
            raise ValueError("responses must be a 2-D persons x n_items array")
        if y.size > MAX_SCORE_CELLS:
            raise ValueError(
                f"response matrix exceeds the {MAX_SCORE_CELLS}-cell scoring limit"
            )
    output_cells = int(y.shape[0]) * draw_count * int(bundle["n_dims"])
    if output_cells > MAX_SERVING_OUTPUT_CELLS:
        raise ValueError(
            f"plausible-values output size ({output_cells} cells) exceeds the "
            f"{MAX_SERVING_OUTPUT_CELLS}-cell serving limit"
        )
    observed = ~np.isnan(y)
    obs_vals = y[observed]
    if obs_vals.size and not np.all((obs_vals == 0.0) | (obs_vals == 1.0)):
        raise ValueError("observed responses must be 0 or 1")
    mean, sd = serving_prior(bundle) if prior is None else (
        np.asarray(prior[0], dtype=float), np.asarray(prior[1], dtype=float))
    pv = core.plausible_values(
        np.where(observed, y, 0.0).ravel(), observed.ravel(), int(y.shape[0]),
        prior_mean=mean, prior_sd=sd,
        q_theta=int(bundle["quadrature"]["q_theta"]), xi_rule="gh",
        q_xi=int(bundle["quadrature"]["q_xi"]), n_draws=draw_count, seed=int(seed),
        **_bundle_bank_args(bundle),
    )
    return np.asarray(pv).reshape(y.shape[0], draw_count, bundle["n_dims"])
