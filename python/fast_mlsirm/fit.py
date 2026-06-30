from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .config import FitConfig
from .math import logit, normalize_latent_positions, standardize
from .objective import model_flags, neg_loglik_and_grad, prepare_response, validate_factor_id
from .types import FitResult, MLSIRMParams


def fit(
    responses: np.ndarray,
    factor_id: np.ndarray,
    config: FitConfig | None = None,
    mask: np.ndarray | None = None,
) -> FitResult:
    config = config or FitConfig()
    config.validate()
    model = config.normalized_model()

    y, observed = prepare_response(responses, mask)
    _, n_items = y.shape
    factors = np.asarray(factor_id, dtype=np.int64)
    n_dims = 1 if model in {"ULS2PLM", "ULSRM"} else int(factors.max()) + 1
    if model in {"ULS2PLM", "ULSRM"}:
        factors = np.zeros_like(factors)
    factors = validate_factor_id(factors, n_items, n_dims)

    best: FitResult | None = None
    for restart in range(config.n_restarts):
        candidate = _fit_single_restart(restart, config, y, observed, factors, n_dims, model)
        if best is None or candidate.objective < best.objective:
            best = candidate

    if best is None:
        raise RuntimeError("Optimization failed to find a valid fit.")
    return best


def _fit_single_restart(
    restart: int,
    config: FitConfig,
    y: np.ndarray,
    observed: np.ndarray,
    factors: np.ndarray,
    n_dims: int,
    model: str,
) -> FitResult:
    rng = np.random.default_rng(config.seed + restart)
    params0 = _initial_params(y, observed, factors, n_dims, config.latent_dim, config, rng)
    x0 = _pack(params0, model)
    objective = _make_objective(y, observed, factors, params0, config)

    x = x0
    obj_trace: list[float] = []
    loglik_trace: list[float] = []
    status = "max_iter_reached"
    n_iter = 0

    if config.optimizer in {"adam", "adam_lbfgs"}:
        adam_iter = config.max_iter if config.optimizer == "adam" else max(1, config.max_iter // 2)
        x, adam_obj, adam_loglik, status = _adam(x, objective, config, adam_iter)
        obj_trace.extend(adam_obj)
        loglik_trace.extend(adam_loglik)
        n_iter += len(adam_obj)

    if config.optimizer in {"lbfgs", "adam_lbfgs"}:
        lbfgs_iter = config.max_iter if config.optimizer == "lbfgs" else max(1, config.max_iter - n_iter)
        x, lbfgs_obj, lbfgs_loglik, status = _lbfgs(x, objective, config, lbfgs_iter)
        obj_trace.extend(lbfgs_obj)
        loglik_trace.extend(lbfgs_loglik)
        n_iter += len(lbfgs_obj)

    final_params = _unpack(x, params0, model)
    if model != "MIRT":
        final_params = normalize_latent_positions(final_params)
    final_obj, _, final_loglik = neg_loglik_and_grad(y, factors, final_params, config, mask=observed)
    obj_trace.append(final_obj)
    loglik_trace.append(final_loglik)

    candidate = FitResult(
        params=final_params,
        model=model,
        optimizer=config.optimizer,
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
    for d in range(n_dims):
        items = factor_id == d
        denom = np.maximum(observed[:, items].sum(axis=1), 1)
        theta[:, d] = standardize((y[:, items] * observed[:, items]).sum(axis=1) / denom)

    item_counts = np.maximum(observed.sum(axis=0), 1)
    item_means = (y * observed).sum(axis=0) / item_counts
    b = logit(item_means)
    alpha = rng.normal(0.0, 0.02, size=n_items)

    xi = rng.normal(0.0, 0.1, size=(n_persons, latent_dim))
    zeta = rng.normal(0.0, 0.1, size=(n_items, latent_dim))
    tau = float(np.log(config.init_gamma))
    return normalize_latent_positions(MLSIRMParams(theta=theta, alpha=alpha, b=b, xi=xi, zeta=zeta, tau=tau))


def _make_objective(
    y: np.ndarray,
    observed: np.ndarray,
    factor_id: np.ndarray,
    template: MLSIRMParams,
    config: FitConfig,
) -> Callable[[np.ndarray], tuple[float, np.ndarray, float]]:
    model = config.normalized_model()

    def objective(x: np.ndarray) -> tuple[float, np.ndarray, float]:
        params = _unpack(x, template, model)
        obj, grad, loglik = neg_loglik_and_grad(y, factor_id, params, config, mask=observed)
        grad_vec = _pack(grad, model)
        if config.gradient_clip is not None:
            norm = float(np.linalg.norm(grad_vec))
            if norm > config.gradient_clip:
                grad_vec = grad_vec * (config.gradient_clip / norm)
        return obj, grad_vec, loglik

    return objective


def _pack(params: MLSIRMParams, model: str) -> np.ndarray:
    free_alpha, uses_space = model_flags(model)
    parts = [params.theta.ravel()]
    if free_alpha:
        parts.append(params.alpha.ravel())
    parts.append(params.b.ravel())
    if uses_space:
        parts.extend([params.xi.ravel(), params.zeta.ravel(), np.array([params.tau], dtype=np.float64)])
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
        alpha = np.zeros_like(template.alpha)

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

    return MLSIRMParams(theta=np.array(theta), alpha=np.array(alpha), b=np.array(b), xi=np.array(xi), zeta=np.array(zeta), tau=tau)


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
            return x, trace, loglik_trace, "nan_or_inf"
        trace.append(float(obj))
        loglik_trace.append(float(loglik))
        if abs(prev - obj) / max(1.0, abs(prev)) < config.tolerance:
            status = "converged"
            break
        prev = obj
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        x -= config.learning_rate * (m / (1.0 - beta1**t)) / (np.sqrt(v / (1.0 - beta2**t)) + 1e-8)
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
            status = "converged"
            break

        direction = -_lbfgs_direction(grad, s_hist, y_hist, rho_hist)
        if float(np.dot(grad, direction)) >= 0:
            direction = -grad

        step = 1.0
        slope = float(np.dot(grad, direction))
        accepted = False
        for _line in range(20):
            candidate = x + step * direction
            next_obj, next_grad, next_loglik = objective(candidate)
            if np.isfinite(next_obj) and next_obj <= obj + 1e-4 * step * slope:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            status = "line_search_failed"
            break

        s = candidate - x
        y_delta = next_grad - grad
        ys = float(np.dot(y_delta, s))
        if ys > 1e-12:
            s_hist.append(s)
            y_hist.append(y_delta)
            rho_hist.append(1.0 / ys)
            if len(s_hist) > config.lbfgs_history:
                s_hist.pop(0)
                y_hist.pop(0)
                rho_hist.pop(0)

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
        alpha = rho * float(np.dot(s, q))
        alphas.append(alpha)
        q -= alpha * y

    if s_hist:
        sy = float(np.dot(s_hist[-1], y_hist[-1]))
        yy = float(np.dot(y_hist[-1], y_hist[-1]))
        q *= sy / yy if yy > 1e-12 else 1.0

    for s, y, rho, alpha in zip(s_hist, y_hist, rho_hist, reversed(alphas)):
        beta = rho * float(np.dot(y, q))
        q += s * (alpha - beta)
    return q
