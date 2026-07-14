from __future__ import annotations

import numpy as np

from .config import FitConfig
from .fit import _pack, _unpack
from .objective import neg_loglik_and_grad
from .types import MLSIRMParams


def observed_information(
    responses: np.ndarray,
    factor_id: np.ndarray,
    params: MLSIRMParams,
    config: FitConfig | None = None,
    mask: np.ndarray | None = None,
    backend: str | None = None,
    device: str | None = "cpu",
    step: float = 1e-4,
) -> np.ndarray:
    """Finite-difference Hessian of the penalized negative log-likelihood.

    The default Rust device is CPU so finite-difference curvature uses the f64
    path even when model fitting defaults to ``rust_device="auto"`` on GPU hosts.
    Pass ``device=None`` to honor ``config.rust_device`` instead.
    """
    config = config or FitConfig()
    model = config.normalized_model()
    chosen_backend = config.backend if backend is None else backend
    x0 = _pack(params, model)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be > 0 and finite")

    def objective(x: np.ndarray) -> float:
        value, _, _ = neg_loglik_and_grad(
            responses,
            factor_id,
            _unpack(x, params, model),
            config=config,
            mask=mask,
            backend=chosen_backend,
            device=device,
        )
        if not np.isfinite(value):
            raise ValueError("objective must be finite for Hessian calculation")
        return float(value)

    n = x0.size
    MAX_HESSIAN_DIM = 5_000
    if n > MAX_HESSIAN_DIM:
        raise ValueError(
            f"observed_information supports at most {MAX_HESSIAN_DIM} parameters (got {n}); "
            "the dense finite-difference Hessian is O(n^2) memory and O(n^2) objective calls"
        )
    hessian = np.zeros((n, n), dtype=np.float64)
    base = objective(x0)
    eye = np.eye(n, dtype=np.float64)
    h = float(step)

    for i in range(n):
        x_plus = x0 + h * eye[i]
        x_minus = x0 - h * eye[i]
        hessian[i, i] = (objective(x_plus) - 2.0 * base + objective(x_minus)) / (h * h)
        for j in range(i + 1, n):
            f_pp = objective(x0 + h * eye[i] + h * eye[j])
            f_pm = objective(x0 + h * eye[i] - h * eye[j])
            f_mp = objective(x0 - h * eye[i] + h * eye[j])
            f_mm = objective(x0 - h * eye[i] - h * eye[j])
            value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h)
            hessian[i, j] = value
            hessian[j, i] = value

    return (hessian + hessian.T) / 2.0


def second_order_test(hessian: np.ndarray, tol: float = 1e-8) -> dict[str, float | bool | np.ndarray]:
    """Check whether the Hessian/information matrix is positive definite."""
    matrix = np.asarray(hessian, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("hessian must be a square matrix")
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
    min_eigenvalue = float(eigenvalues.min()) if eigenvalues.size else float("nan")
    return {
        "passed": bool(np.all(eigenvalues > tol)),
        "min_eigenvalue": min_eigenvalue,
        "eigenvalues": eigenvalues,
    }


def vcov_from_hessian(hessian: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """Invert the observed information, falling back to a Moore-Penrose inverse."""
    matrix = np.asarray(hessian, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("hessian must be a square matrix")
    try:
        vcov = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        vcov = np.linalg.pinv(matrix, rcond=rcond)
    return (vcov + vcov.T) / 2.0


def standard_errors_from_vcov(vcov: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vcov, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("vcov must be a square matrix")
    return np.sqrt(np.maximum(np.diag(matrix), 0.0))


def oakes_standard_errors(
    result,
    responses,
    factor_id,
    config=None,
    mask=None,
    group_id=None,
    cluster_id=None,
    h: float = 1e-5,
) -> dict:
    """Item-parameter standard errors for a marginal (MMLE) fit via Oakes'
    identity — the estimator Pritikin (2017) recommends in the EM framework
    (M-step Hessian at the fixed posterior plus a finite-differenced cross
    term, one E-step per parameter). Population parameters are conditioned
    on; anchors/zero-inflation/covariates are not supported. Runs on the CPU
    in f64 (finite differences would drown in f32 GPU noise).

    Returns ``{"labels", "se", "information"}`` with labels ``alpha:i``,
    ``b:i``, ``zeta:i:k``, ``tau``.
    """
    import numpy as np

    from . import _core
    from .config import FitConfig
    from .estimators.marginal import LSIRM_PRIOR
    from .objective import prepare_response

    config = config or FitConfig(model=result.model, estimator="mmle")
    y, observed = prepare_response(np.asarray(responses, dtype=float), mask)
    n_persons, n_items = y.shape
    raw_factors = np.asarray(factor_id)
    if raw_factors.ndim != 1 or raw_factors.shape != (n_items,):
        raise ValueError("factor_id must be a 1-D array with one entry per item")
    ff = raw_factors.astype(np.float64)
    if not np.all(np.isfinite(ff)) or np.any(ff < 0) or np.any(ff != np.floor(ff)):
        raise ValueError("factor_id must be finite non-negative integers")
    factors = raw_factors.astype(np.int64)
    n_dims = int(factors.max()) + 1 if factors.size else 0
    if n_dims > n_items:
        raise ValueError("factor_id implies more dimensions than items")
    pop = result.population or {}
    from .fit import _compact_population_labels
    if group_id is not None:
        ids, n_pop = _compact_population_labels(group_id, n_persons, "group_id")
        pop_kind = "multigroup"
    elif cluster_id is not None:
        ids, n_pop = _compact_population_labels(cluster_id, n_persons, "cluster_id")
        pop_kind = "multilevel"
    else:
        ids, pop_kind, n_pop = None, "single", 0
    mu = np.asarray(pop.get("mu", np.zeros((0,))), dtype=np.float64).ravel()
    sigma = np.asarray(pop.get("sigma", np.ones((0,))), dtype=np.float64).ravel()
    sigma_u = float(pop.get("sigma_u", 0.0))
    p = result.params
    return dict(
        _core.oakes_standard_errors(
            np.where(observed, y, 0.0).ravel(),
            observed.ravel(),
            factors,
            int(n_persons),
            int(n_items),
            int(n_dims),
            int(np.asarray(p.zeta).shape[1]),
            result.model,
            float(config.eps_distance),
            np.asarray(p.alpha, dtype=np.float64),
            np.asarray(p.b, dtype=np.float64),
            np.asarray(p.zeta, dtype=np.float64).ravel(),
            float(p.tau),
            pop_kind=pop_kind,
            pop_id=ids,
            n_pop=int(n_pop),
            mu=mu if mu.size else None,
            sigma=sigma if sigma.size else None,
            sigma_u=sigma_u,
            q_theta=int(config.q_theta),
            q_xi=int(config.q_xi),
            q_u=int(config.q_u),
            xi_rule=config.xi_rule,
            xi_points=int(config.xi_points),
            xi_seed=int(config.xi_seed),
            lambda_b=LSIRM_PRIOR["lambda_b"],
            lambda_alpha=LSIRM_PRIOR["lambda_alpha"],
            mu_alpha=LSIRM_PRIOR["mu_alpha"],
            lambda_zeta=LSIRM_PRIOR["lambda_zeta"],
            lambda_tau=LSIRM_PRIOR["lambda_tau"],
            mu_tau=LSIRM_PRIOR["mu_tau"],
            h=float(h),
        )
    )
