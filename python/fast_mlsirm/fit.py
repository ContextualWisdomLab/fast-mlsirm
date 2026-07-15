from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .backend import normalize_device, resolve_backend
from .config import FitConfig, PenaltyConfig
from .math import logit, normalize_latent_positions, standardize
from .objective import (model_flags, neg_loglik_and_grad, prepare_response,
                        validate_factor_id)
from .types import FitResult, MLSIRMParams


def _compact_population_labels(raw, n_persons: int, name: str):
    """Validate and compact caller-supplied population labels to contiguous
    ``0..k-1`` ids. Rejects non-1-D, wrong-length, non-finite, non-integer, or
    negative labels, and remaps the observed labels so the derived group/cluster
    count is the number of *distinct* labels (<= n_persons) rather than
    ``max(label) + 1`` -- which otherwise lets sparse ids such as ``[0, 1e9]``
    force unbounded population allocations (memory-exhaustion DoS)."""
    import numpy as _np

    arr = _np.asarray(raw)
    if arr.ndim != 1 or arr.shape[0] != n_persons:
        raise ValueError(f"{name} must be a 1-D array of length n_persons ({n_persons})")
    fl = arr.astype(_np.float64)
    if not _np.all(_np.isfinite(fl)):
        raise ValueError(f"{name} must be finite")
    if _np.any(fl < 0) or _np.any(fl != _np.floor(fl)):
        raise ValueError(f"{name} must be non-negative integers")
    uniq, remapped = _np.unique(arr.astype(_np.int64), return_inverse=True)
    return remapped.astype(_np.int64), int(uniq.size)


def fit(
    responses: np.ndarray,
    factor_id: np.ndarray,
    config: FitConfig | None = None,
    mask: np.ndarray | None = None,
    group_id: np.ndarray | None = None,
    cluster_id: np.ndarray | None = None,
    anchors: dict | None = None,
    covariate: dict | None = None,
) -> FitResult:
    """Fit a latent-space model.

    ``covariate`` = ``{"w": (n_groups x n_items) array, "init_delta": float}``
    switches on a context-varying item covariate with one estimated
    coefficient (Debeer & Janssen 2013 linear item-position effect):
    ``eta += delta * w[group(p), i]``. Requires multigroup contexts (booklets)
    or anchors for identification.

    ``group_id``/``cluster_id`` (mutually exclusive, ``estimator="mmle"`` only)
    switch on estimation-level population structures: multigroup calibration
    (Bock & Zimowski 1997 — group-specific trait means/SDs, common items,
    group 0 as the N(0,1) reference) and multilevel random intercepts
    (Fox & Glas 2001 — cluster intercept SD ``sigma_u`` estimated).

    ``anchors`` enables Fixed Item Parameter Calibration (Kim 2006, the
    MWU-MEM-style variant): ``{"fixed": bool[I], "alpha", "b", "zeta",
    "tau" (optional)}``. Anchored items stay frozen; without a
    ``group_id``/``cluster_id`` the population mean/SD is freed
    (concurrent-calibration-ready ``singlefree`` population).
    """
    config = config or FitConfig()
    config.validate()
    backend = resolve_backend(config.backend)
    # The device is a sub-option of the rust backend; the numpy path ignores it.
    device = normalize_device(config.rust_device) if backend == "rust" else "cpu"
    model = config.normalized_model()

    y, observed = prepare_response(responses, mask)
    _, n_items = y.shape
    factors = np.asarray(factor_id, dtype=np.int64)
    n_dims = 1 if model in {"ULS2PLM", "ULSRM"} else int(factors.max()) + 1
    if n_dims > n_items:
        raise ValueError("factor_id implies more dimensions than items")

    if model in {"ULS2PLM", "ULSRM"}:
        factors = np.zeros_like(factors)  # pragma: no cover
    factors = validate_factor_id(factors, n_items, n_dims)

    if group_id is not None and cluster_id is not None:
        raise ValueError("group_id and cluster_id are mutually exclusive")
    if (group_id is not None or cluster_id is not None) and config.estimator != "mmle":
        raise ValueError(
            "estimation-level multigroup/multilevel structures require estimator='mmle'"
        )
    if anchors is not None and config.estimator != "mmle":
        raise ValueError("anchors (FIPC) require estimator='mmle'")
    if anchors is not None and cluster_id is not None:
        raise ValueError("anchors with a multilevel structure are not supported yet")
    if covariate is not None and config.estimator != "mmle":
        raise ValueError("item covariates require estimator='mmle'")
    if covariate is not None and cluster_id is not None:
        raise ValueError("item covariates with a multilevel structure are not supported")

    if model == "BIFAC2PLM" and config.estimator != "mmle":
        raise NotImplementedError(
            "BIFAC2PLM (bifactor) is supported by the marginal estimator only; "
            "use estimator='mmle'."
        )
    if config.estimator == "mmle":
        if (
            model in {"ULS2PLM", "ULSRM"}
            and group_id is None
            and cluster_id is None
            and anchors is None
            and covariate is None
            and not config.zero_inflation
        ):
            # Legacy fast path: plain unidimensional 2PL margin (the latent
            # space is not estimated — unchanged public behavior). Use the
            # spatial models or a population structure for the full marginal
            # latent-space fit.
            return _fit_mmle(y, observed, model, config)
        return _fit_mmle_marginal(
            y, observed, factors, n_dims, model, config, backend, device,
            group_id=group_id, cluster_id=cluster_id, anchors=anchors,
            covariate=covariate,
        )
    if config.estimator in {"em", "bayes"}:
        raise NotImplementedError(
            f"estimator '{config.estimator}' is reserved for a future milestone; "
            "use 'jmle' or 'mmle'."
        )

    best: FitResult | None = None
    for restart in range(config.n_restarts):
        candidate = _run_single_fit(
            y, observed, factors, n_dims, config, model, restart, backend, device
        )
        if best is None or candidate.objective < best.objective:
            best = candidate

    if best is None:
        raise RuntimeError("Optimization failed to find a valid fit.")  # pragma: no cover
    return best


def _fit_mmle(
    y: np.ndarray,
    observed: np.ndarray,
    model: str,
    config: FitConfig,
) -> FitResult:
    """Marginal MLE (EM) — robust to missing data. Unidimensional 2PL measurement.

    Uses the compiled Rust core (``fast_mlsirm._core.fit_mmle_2pl``) when present;
    otherwise the pure-numpy reference in ``fast_mlsirm.estimators.mmle``. Both
    integrate ability over Gauss-Hermite quadrature, so unanswered items
    contribute nothing (missing-at-random safe) — no imputation.
    """
    from .estimators.mmle import fit_mmle_2pl as _py_mmle

    n_persons, n_items = y.shape
    y_filled = np.where(observed, y, 0.0).astype(np.float64)

    rust = None
    try:  # pragma: no cover - depends on the compiled extension being built
        from . import _core  # type: ignore

        rust = getattr(_core, "fit_mmle_2pl", None)
    except Exception:  # pragma: no cover
        rust = None

    if rust is not None:  # pragma: no cover - exercised only when the ext is built
        a, b, theta, loglik_trace, converged = rust(
            y_filled.ravel(),
            observed.astype(bool).ravel(),
            int(n_persons),
            int(n_items),
            int(config.max_iter),
            float(config.tolerance),
        )
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        theta = np.asarray(theta, dtype=np.float64)
        loglik_trace = list(loglik_trace)
    else:
        res = _py_mmle(
            y_filled,
            observed.astype(bool),
            max_iter=config.max_iter,
            tol=config.tolerance,
        )
        a = np.asarray(res["a"], dtype=np.float64)
        b = np.asarray(res["b"], dtype=np.float64)
        theta = np.asarray(res["theta"], dtype=np.float64)
        loglik_trace = list(res["loglik_trace"])
        converged = res["status"] == "converged"

    params = MLSIRMParams(
        theta=theta.reshape(n_persons, 1),
        alpha=np.log(np.clip(a, 1e-6, None)),  # model stores alpha; a = exp(alpha)
        b=b,
        xi=np.zeros((n_persons, config.latent_dim)),
        zeta=np.zeros((n_items, config.latent_dim)),
        tau=0.0,
    )
    return FitResult(
        params=params,
        model=model,
        optimizer=f"mmle_em/{'rust' if rust is not None else 'numpy'}",
        backend=config.backend,
        rust_device=config.rust_device,
        objective=float(-loglik_trace[-1]) if loglik_trace else float("nan"),
        loglik_trace=[float(v) for v in loglik_trace],
        objective_trace=[float(-v) for v in loglik_trace],
        convergence_status="converged" if converged else "max_iter_reached",
        n_iter=len(loglik_trace),
    )


def _fit_mmle_marginal(
    y: np.ndarray,
    observed: np.ndarray,
    factors: np.ndarray,
    n_dims: int,
    model: str,
    config: FitConfig,
    backend: str,
    device: str,
    group_id: np.ndarray | None = None,
    cluster_id: np.ndarray | None = None,
    anchors: dict | None = None,
    covariate: dict | None = None,
) -> FitResult:
    """Marginal EM for the latent-space family (Rust core, NumPy fallback).

    Person latents are integrated out by Gauss-Hermite quadrature; item-side
    parameters carry the LSIRM priors of Jeon et al. (2021) as MAP penalties
    (see ``estimators/marginal.py`` / ``mlsirm-core/src/marginal.rs``).
    """
    from .estimators.marginal import LSIRM_PRIOR, fit_marginal_numpy

    n_persons, n_items = y.shape
    if group_id is not None:
        ids, n_pop = _compact_population_labels(group_id, n_persons, "group_id")
        pop_kind = "multigroup"
    elif cluster_id is not None:
        ids, n_pop = _compact_population_labels(cluster_id, n_persons, "cluster_id")
        pop_kind = "multilevel"
    elif anchors is not None:
        # FIPC: anchored items identify a free single population.
        ids, pop_kind, n_pop = None, "singlefree", 1
    else:
        ids, pop_kind, n_pop = None, "single", 0
    covariate_kwargs: dict = {}
    if covariate is not None:
        covariate_w = np.asarray(covariate["w"], dtype=np.float64).ravel()
        n_ctx = int(n_pop) if pop_kind == "multigroup" else 1
        if covariate_w.size != n_ctx * n_items:
            raise ValueError(
                f"covariate w must have {n_ctx} x {n_items} entries (n_contexts x n_items)"
            )
        if not np.all(np.isfinite(covariate_w)):
            raise ValueError("covariate w must be finite")
        covariate_kwargs = dict(
            covariate_w=covariate_w,
            covariate_init_delta=float(covariate.get("init_delta", 0.0)),
        )
    anchor_kwargs: dict = {}
    if anchors is not None:
        fixed = np.asarray(anchors["fixed"], dtype=bool)
        a_alpha = np.asarray(anchors["alpha"], dtype=np.float64)
        a_b = np.asarray(anchors["b"], dtype=np.float64)
        a_zeta = np.asarray(anchors["zeta"], dtype=np.float64).ravel()
        if fixed.shape != (n_items,):
            raise ValueError(f"anchor_fixed must have shape ({n_items},)")
        if a_alpha.shape != (n_items,) or a_b.shape != (n_items,):
            raise ValueError(f"anchor alpha/b must have shape ({n_items},)")
        if a_zeta.size != n_items * int(config.latent_dim):
            raise ValueError("anchor zeta must have n_items x latent_dim entries")
        if not (np.all(np.isfinite(a_alpha)) and np.all(np.isfinite(a_b)) and np.all(np.isfinite(a_zeta))):
            raise ValueError("anchor alpha/b/zeta must be finite")
        anchor_kwargs = dict(
            anchor_fixed=fixed,
            anchor_alpha=a_alpha,
            anchor_b=a_b,
            anchor_zeta=a_zeta,
            anchor_tau=None if anchors.get("tau") is None else float(anchors["tau"]),
        )
    if ids is not None:
        if ids.shape != (n_persons,):
            raise ValueError(f"{pop_kind} ids must have shape (n_persons,)")
        if ids.size and ids.min() < 0:
            raise ValueError(f"{pop_kind} ids must be >= 0")

    # MAP penalties: the paper priors, unless the caller customized the
    # penalty config away from its (JML-oriented) defaults.
    pen = dict(LSIRM_PRIOR)
    if config.penalty != PenaltyConfig():
        pen = {
            "lambda_b": config.penalty.lambda_b,
            "lambda_alpha": config.penalty.lambda_alpha,
            "mu_alpha": config.penalty.mu_alpha,
            "lambda_zeta": config.penalty.lambda_zeta,
            "lambda_tau": config.penalty.lambda_tau,
            "mu_tau": config.penalty.mu_tau,
        }

    rust = None
    if backend == "rust":
        try:  # pragma: no cover - depends on the compiled extension
            from . import _core  # type: ignore

            rust = getattr(_core, "fit_marginal", None)
        except Exception:  # pragma: no cover
            rust = None

    y_filled = np.where(observed, y, 0.0).astype(np.float64)
    if rust is not None:  # pragma: no cover - exercised only with the extension
        try:
            res = rust(
                y_filled.ravel(),
                observed.astype(bool).ravel(),
                factors.astype(np.int64),
                int(n_persons),
                int(n_items),
                int(n_dims),
                int(config.latent_dim),
                model,
                float(config.eps_distance),
                pop_kind=pop_kind,
                pop_id=None if ids is None else ids,
                n_pop=int(n_pop),
                q_theta=int(config.q_theta),
                q_xi=int(config.q_xi),
                q_u=int(config.q_u),
                max_iter=int(config.max_iter),
                tol=float(config.tolerance),
                m_steps=int(config.m_steps),
                lambda_b=pen["lambda_b"],
                lambda_alpha=pen["lambda_alpha"],
                mu_alpha=pen["mu_alpha"],
                lambda_zeta=pen["lambda_zeta"],
                lambda_tau=pen["lambda_tau"],
                mu_tau=pen["mu_tau"],
                device=device,
                xi_rule=config.xi_rule,
                xi_points=int(config.xi_points),
                xi_seed=int(config.xi_seed),
                zero_inflation=bool(config.zero_inflation),
                **anchor_kwargs,
                **covariate_kwargs,
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        alpha = np.asarray(res["alpha"], dtype=np.float64)
        b = np.asarray(res["b"], dtype=np.float64)
        zeta = np.asarray(res["zeta"], dtype=np.float64).reshape(
            n_items, config.latent_dim
        )
        tau = float(res["tau"])
        theta_eap = np.asarray(res["theta_eap"], dtype=np.float64).reshape(
            n_persons, n_dims
        )
        theta_sd = np.asarray(res["theta_sd"], dtype=np.float64).reshape(
            n_persons, n_dims
        )
        xi_eap = np.asarray(res["xi_eap"], dtype=np.float64).reshape(
            n_persons, config.latent_dim
        )
        mu = np.asarray(res["mu"], dtype=np.float64).reshape(-1, n_dims)
        sigma = np.asarray(res["sigma"], dtype=np.float64).reshape(-1, n_dims)
        sigma_u = float(res["sigma_u"])
        u_eap = np.asarray(res["u_eap"], dtype=np.float64)
        loglik_trace = [float(v) for v in res["loglik_trace"]]
        n_iter = int(res["n_iter"])
        converged = bool(res["converged"])
        ic = dict(res["ic"]) if "ic" in res else None
        delta = float(res.get("delta", 0.0))
        pi_zero = float(res.get("pi_zero", 0.0))
        zero_resp = np.asarray(res.get("zero_responsibility", []), dtype=np.float64)
        optimizer = "mmle_marginal_em/rust"
    else:
        pop: dict = {"kind": pop_kind}
        if pop_kind == "multigroup":
            pop = {"kind": "multigroup", "group_id": ids, "n_groups": n_pop}
        elif pop_kind == "multilevel":
            pop = {"kind": "multilevel", "cluster_id": ids, "n_clusters": n_pop}
        res = fit_marginal_numpy(
            y_filled,
            observed.astype(bool),
            factors,
            model=model,
            n_dims=n_dims,
            latent_dim=config.latent_dim,
            pop=pop,
            q_theta=config.q_theta,
            q_xi=config.q_xi,
            q_u=config.q_u,
            max_iter=config.max_iter,
            tol=config.tolerance,
            m_steps=config.m_steps,
            eps_distance=config.eps_distance,
            penalty=pen,
            xi_rule=config.xi_rule,
            xi_points=int(config.xi_points),
            xi_seed=int(config.xi_seed),
            anchors=anchors,
            zero_inflation=bool(config.zero_inflation),
            covariate=covariate,
        )
        alpha, b, zeta, tau = res["alpha"], res["b"], res["zeta"], res["tau"]
        theta_eap, theta_sd = res["theta_eap"], res["theta_sd"]
        xi_eap = res["xi_eap"]
        mu, sigma = res["mu"], res["sigma"]
        sigma_u, u_eap = res["sigma_u"], res["u_eap"]
        loglik_trace = [float(v) for v in res["loglik_trace"]]
        n_iter = int(res["n_iter"])
        converged = bool(res["converged"])
        ic = res.get("ic")
        delta = float(res.get("delta", 0.0))
        pi_zero = float(res.get("pi_zero", 0.0))
        zero_resp = np.asarray(res.get("zero_responsibility", []), dtype=np.float64)
        optimizer = "mmle_marginal_em/numpy"

    population: dict = {"kind": pop_kind, "theta_sd": theta_sd}
    if config.zero_inflation:
        population.update(pi_zero=pi_zero, zero_responsibility=zero_resp)
    if covariate is not None:
        population.update(delta=delta)
    if pop_kind in {"multigroup", "singlefree"}:
        population.update(mu=mu, sigma=sigma)
    elif pop_kind == "multilevel":
        icc = sigma_u**2 / (sigma_u**2 + 1.0)
        population.update(sigma_u=sigma_u, u_eap=u_eap, icc=icc)
    if anchors is not None:
        population.update(
            fixed_items=np.asarray(anchors["fixed"], dtype=bool).copy(),
            tau_fixed=anchors.get("tau") is not None,
        )

    params = MLSIRMParams(
        theta=theta_eap,
        alpha=alpha,
        b=b,
        xi=xi_eap,
        zeta=zeta,
        tau=tau,
    )
    return FitResult(
        params=params,
        model=model,
        optimizer=optimizer,
        backend=backend,
        rust_device=device,
        objective=float(-loglik_trace[-1]) if loglik_trace else float("nan"),
        loglik_trace=loglik_trace,
        objective_trace=[float(-v) for v in loglik_trace],
        convergence_status="converged" if converged else "max_iter_reached",
        n_iter=n_iter,
        population=population,
        ic=ic,
    )


def _run_single_fit(
    y: np.ndarray,
    observed: np.ndarray,
    factors: np.ndarray,
    n_dims: int,
    config: FitConfig,
    model: str,
    restart: int,
    backend: str,
    device: str,
) -> FitResult:
    rng = np.random.default_rng(config.seed + restart)
    params0 = _initial_params(
        y, observed, factors, n_dims, config.latent_dim, config, rng
    )
    x0 = _pack(params0, model)
    objective = _make_objective(y, observed, factors, params0, config, backend, device)

    x = x0
    obj_trace: list[float] = []
    loglik_trace: list[float] = []
    status = "max_iter_reached"
    n_iter = 0

    if config.optimizer in {"adam", "adam_lbfgs"}:
        adam_iter = (
            config.max_iter
            if config.optimizer == "adam"
            else max(1, config.max_iter // 2)
        )
        x, adam_obj, adam_loglik, status = _adam(x, objective, config, adam_iter)
        obj_trace.extend(adam_obj)
        loglik_trace.extend(adam_loglik)
        n_iter += len(adam_obj)

    if config.optimizer in {"lbfgs", "adam_lbfgs"}:
        lbfgs_iter = (
            config.max_iter
            if config.optimizer == "lbfgs"
            else max(1, config.max_iter - n_iter)
        )
        x, lbfgs_obj, lbfgs_loglik, status = _lbfgs(x, objective, config, lbfgs_iter)
        obj_trace.extend(lbfgs_obj)
        loglik_trace.extend(lbfgs_loglik)
        n_iter += len(lbfgs_obj)

    final_params = _unpack(x, params0, model)
    if model != "MIRT":
        final_params = normalize_latent_positions(final_params)
    final_obj, _, final_loglik = neg_loglik_and_grad(
        y, factors, final_params, config, mask=observed, backend=backend, device=device
    )
    obj_trace.append(final_obj)
    loglik_trace.append(final_loglik)

    candidate = FitResult(
        params=final_params,
        model=model,
        optimizer=config.optimizer,
        backend=backend,
        rust_device=device,
        objective=final_obj,
        loglik_trace=loglik_trace,
        objective_trace=obj_trace,
        convergence_status=status,
        n_iter=n_iter,
    )
    return candidate


def _initial_params(
    y: np.ndarray,
    observed: np.ndarray,
    factor_id: np.ndarray,
    n_dims: int,
    latent_dim: int,
    config: FitConfig,
    rng: np.random.Generator,
) -> MLSIRMParams:
    n_persons, n_items = y.shape
    theta = np.zeros((n_persons, n_dims), dtype=np.float64)

    item_mask = factor_id[:, None] == np.arange(n_dims)
    denom = np.maximum(observed @ item_mask.astype(np.float64), 1)
    x = ((y * observed) @ item_mask.astype(np.float64)) / denom

    for d in range(n_dims):
        theta[:, d] = standardize(x[:, d])

    item_counts = np.maximum(observed.sum(axis=0), 1)
    item_means = (y * observed).sum(axis=0) / item_counts
    b = logit(item_means)
    alpha = rng.normal(0.0, 0.02, size=n_items)

    xi = rng.normal(0.0, 0.1, size=(n_persons, latent_dim))
    zeta = rng.normal(0.0, 0.1, size=(n_items, latent_dim))
    tau = float(np.log(config.init_gamma))
    return normalize_latent_positions(
        MLSIRMParams(theta=theta, alpha=alpha, b=b, xi=xi, zeta=zeta, tau=tau)
    )


def _make_objective(
    y: np.ndarray,
    observed: np.ndarray,
    factor_id: np.ndarray,
    template: MLSIRMParams,
    config: FitConfig,
    backend: str,
    device: str,
) -> Callable[[np.ndarray], tuple[float, np.ndarray, float]]:
    model = config.normalized_model()

    def objective(x: np.ndarray) -> tuple[float, np.ndarray, float]:
        params = _unpack(x, template, model)
        obj, grad, loglik = neg_loglik_and_grad(
            y, factor_id, params, config, mask=observed, backend=backend, device=device
        )
        grad_vec = _pack(grad, model)
        if config.gradient_clip is not None:
            norm = float(np.linalg.norm(grad_vec))
            if norm > config.gradient_clip:
                grad_vec = grad_vec * (config.gradient_clip / norm)  # pragma: no cover
        return obj, grad_vec, loglik

    return objective


def _pack(params: MLSIRMParams, model: str) -> np.ndarray:
    free_alpha, uses_space = model_flags(model)
    parts = [params.theta.ravel()]
    if free_alpha:
        parts.append(params.alpha.ravel())
    parts.append(params.b.ravel())
    if uses_space:
        parts.extend(
            [
                params.xi.ravel(),
                params.zeta.ravel(),
                np.array([params.tau], dtype=np.float64),
            ]
        )
    return np.concatenate(parts).astype(np.float64, copy=False)


def _unpack(x: np.ndarray, template: MLSIRMParams, model: str) -> MLSIRMParams:
    free_alpha, uses_space = model_flags(model)
    cursor = 0

    theta_size = template.theta.size
    theta = x[cursor : cursor + theta_size].reshape(template.theta.shape)
    cursor += theta_size

    if free_alpha:
        alpha = x[cursor : cursor + template.alpha.size]
        cursor += template.alpha.size
    else:
        alpha = np.zeros_like(template.alpha)  # pragma: no cover

    b = x[cursor : cursor + template.b.size]
    cursor += template.b.size

    if uses_space:
        xi_size = template.xi.size
        zeta_size = template.zeta.size
        xi = x[cursor : cursor + xi_size].reshape(template.xi.shape)
        cursor += xi_size
        zeta = x[cursor : cursor + zeta_size].reshape(template.zeta.shape)
        cursor += zeta_size
        tau = float(x[cursor])
    else:
        xi = np.zeros_like(template.xi)
        zeta = np.zeros_like(template.zeta)
        tau = -30.0

    return MLSIRMParams(
        theta=np.array(theta),
        alpha=np.array(alpha),
        b=np.array(b),
        xi=np.array(xi),
        zeta=np.array(zeta),
        tau=tau,
    )


def _adam(
    x0: np.ndarray,
    objective: Callable[[np.ndarray], tuple[float, np.ndarray, float]],
    config: FitConfig,
    max_iter: int,
) -> tuple[np.ndarray, list[float], list[float], str]:
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    beta1 = 0.9
    beta2 = 0.999
    trace: list[float] = []
    loglik_trace: list[float] = []
    status = "max_iter_reached"
    prev = np.inf

    for t in range(1, max_iter + 1):
        obj, grad, loglik = objective(x)
        if not np.isfinite(obj) or not np.all(np.isfinite(grad)):
            return x, trace, loglik_trace, "nan_or_inf"  # pragma: no cover
        trace.append(float(obj))
        loglik_trace.append(float(loglik))
        if abs(prev - obj) / max(1.0, abs(prev)) < config.tolerance:
            status = "converged"  # pragma: no cover
            break  # pragma: no cover
        prev = obj
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        x -= (
            config.learning_rate
            * (m / (1.0 - beta1**t))
            / (np.sqrt(v / (1.0 - beta2**t)) + 1e-8)
        )
    return x, trace, loglik_trace, status


def _lbfgs(
    x0: np.ndarray,
    objective: Callable[[np.ndarray], tuple[float, np.ndarray, float]],
    config: FitConfig,
    max_iter: int,
) -> tuple[np.ndarray, list[float], list[float], str]:
    x = x0.copy()
    obj, grad, loglik = objective(x)
    trace = [float(obj)]
    loglik_trace = [float(loglik)]
    s_hist: list[np.ndarray] = []
    y_hist: list[np.ndarray] = []
    rho_hist: list[float] = []
    status = "max_iter_reached"

    for _ in range(max_iter):
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm < config.tolerance:
            status = "converged"  # pragma: no cover
            break  # pragma: no cover

        direction = -_lbfgs_direction(grad, s_hist, y_hist, rho_hist)
        if float(np.dot(grad, direction)) >= 0:
            direction = -grad  # pragma: no cover

        step = 1.0
        slope = float(np.dot(grad, direction))
        accepted = False
        for _line in range(20):
            candidate = x + step * direction
            next_obj, next_grad, next_loglik = objective(candidate)
            if np.isfinite(next_obj) and next_obj <= obj + 1e-4 * step * slope:
                accepted = True
                break
            step *= 0.5  # pragma: no cover
        if not accepted:
            status = "line_search_failed"  # pragma: no cover
            break  # pragma: no cover

        s = candidate - x
        y_delta = next_grad - grad
        ys = float(np.dot(y_delta, s))
        if ys > 1e-12:
            s_hist.append(s)
            y_hist.append(y_delta)
            rho_hist.append(1.0 / ys)
            if len(s_hist) > config.lbfgs_history:
                s_hist.pop(0)  # pragma: no cover
                y_hist.pop(0)  # pragma: no cover
                rho_hist.pop(0)  # pragma: no cover

        x, obj, grad, loglik = candidate, next_obj, next_grad, next_loglik
        trace.append(float(obj))
        loglik_trace.append(float(loglik))
    return x, trace, loglik_trace, status


def _lbfgs_direction(
    grad: np.ndarray,
    s_hist: list[np.ndarray],
    y_hist: list[np.ndarray],
    rho_hist: list[float],
) -> np.ndarray:
    q = grad.copy()
    alphas: list[float] = []
    for s, y, rho in zip(reversed(s_hist), reversed(y_hist), reversed(rho_hist)):
        alpha = rho * float(np.dot(s, q))  # pragma: no cover
        alphas.append(alpha)  # pragma: no cover
        q -= alpha * y  # pragma: no cover

    if s_hist:
        sy = float(np.dot(s_hist[-1], y_hist[-1]))  # pragma: no cover
        yy = float(np.dot(y_hist[-1], y_hist[-1]))  # pragma: no cover
        q *= sy / yy if yy > 1e-12 else 1.0  # pragma: no cover

    for s, y, rho, alpha in zip(s_hist, y_hist, rho_hist, reversed(alphas)):
        beta = rho * float(np.dot(y, q))  # pragma: no cover
        q += s * (alpha - beta)  # pragma: no cover
    return q
