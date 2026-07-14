"""NumPy reference for the marginal (MMLE-EM) latent-space estimator.

Mirror of ``crates/mlsirm-core/src/marginal.rs`` — same quadrature tables
(``numpy.polynomial.hermite_e.hermegauss`` with weights normalized to sum 1,
the convention the Rust consts were generated from), same E-step/M-step
algebra, same deterministic initialization and PCA alignment. Kept for parity
testing and as the fallback when the compiled core is unavailable; any change
here must be mirrored in the Rust core (and vice versa).
"""

from __future__ import annotations

import numpy as np

SUPPORTED_Q = (7, 11, 15, 21, 31, 41)

# Priors of Jeon et al. (2021) / lsirm12pl, used as MAP penalties by the
# marginal estimator (mirror of PenaltyConfig::lsirm_prior in Rust):
# beta ~ N(0, 4), log alpha ~ N(0.5, 1), zeta ~ MVN(0, I), log gamma ~ N(0.5, 1).
LSIRM_PRIOR = {
    "lambda_b": 0.25,
    "lambda_alpha": 1.0,
    "mu_alpha": 0.5,
    "lambda_zeta": 1.0,
    "lambda_tau": 1.0,
    "mu_tau": 0.5,
}


def _gh(q: int) -> tuple[np.ndarray, np.ndarray]:
    if q not in SUPPORTED_Q:
        raise ValueError(f"unsupported quadrature size {q}; supported: {SUPPORTED_Q}")
    nodes, weights = np.polynomial.hermite_e.hermegauss(q)
    return nodes, weights / weights.sum()


def _model_flags(model: str) -> tuple[bool, bool]:
    model = model.upper()
    free_alpha = model not in {"MLSRM", "ULSRM"}
    uses_space = model != "MIRT"
    return free_alpha, uses_space


def _interaction_kind(model: str) -> str:
    """Mirror of mlsirm_core::interaction_kind: none | distance | inner."""
    model = model.upper()
    if model == "MIRT":
        return "none"
    if model == "BIFAC2PLM":
        return "inner"
    return "distance"


_HALTON_PRIMES = (2, 3, 5, 7, 11, 13)

# Acklam's inverse normal CDF (same coefficients as the Rust core; parity).
_ACK_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_ACK_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01)
_ACK_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
          -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_ACK_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00)


def _inv_normal_cdf(p: float) -> float:
    a, b, c, d = _ACK_A, _ACK_B, _ACK_C, _ACK_D
    p_low = 0.02425
    if p < p_low:
        q = np.sqrt(-2.0 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= 1.0 - p_low:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = np.sqrt(-2.0 * np.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def _radical_inverse(i: int, base: int) -> float:
    inv, f = 0.0, 1.0 / base
    while i > 0:
        inv += (i % base) * f
        i //= base
        f /= base
    return inv


def _lcg_next(state: int) -> int:
    return (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)


def _lcg_uniform(state: int) -> tuple[float, int]:
    state = _lcg_next(state)
    return (state >> 11) / float(1 << 53), state


def _normal_draw(state: int) -> tuple[float, int]:
    u1, state = _lcg_uniform(state)
    u2, state = _lcg_uniform(state)
    return float(np.sqrt(-2.0 * np.log(max(u1, 1e-12))) * np.cos(2.0 * np.pi * u2)), state


def _xi_nodes(
    rule: str, latent_dim: int, q_xi: int, xi_points: int, xi_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror of ``mlsirm_core::nodes::build_xi_nodes``."""
    rule = rule.lower()
    if rule in {"gh", "gauss-hermite", "gausshermite"}:
        if latent_dim > 3:
            raise ValueError(
                "tensor Gauss-Hermite supports latent_dim <= 3; use xi_rule qmc/mc"
            )
        return _xi_grid(q_xi, latent_dim)
    if rule in {"qmc", "halton"}:
        if xi_points < 1:
            raise ValueError("xi_points must be >= 1 for the Halton/MonteCarlo rules")
        if latent_dim > len(_HALTON_PRIMES):
            raise ValueError(f"Halton rule supports latent_dim <= {len(_HALTON_PRIMES)}")
        shift = np.zeros(latent_dim)
        if xi_seed != 0:
            state = xi_seed
            for k in range(latent_dim):
                shift[k], state = _lcg_uniform(state)
        grid = np.empty((xi_points, latent_dim))
        for j in range(xi_points):
            for k in range(latent_dim):
                u = _radical_inverse(j + 1, _HALTON_PRIMES[k]) + shift[k]
                if u >= 1.0:
                    u -= 1.0
                grid[j, k] = _inv_normal_cdf(min(max(u, 1e-12), 1.0 - 1e-12))
        return grid, np.full(xi_points, -np.log(xi_points))
    if rule in {"mc", "montecarlo", "monte-carlo"}:
        if xi_points < 1:
            raise ValueError("xi_points must be >= 1 for the Halton/MonteCarlo rules")
        state = max(xi_seed, 1)
        grid = np.empty((xi_points, latent_dim))
        for j in range(xi_points):
            for k in range(latent_dim):
                grid[j, k], state = _normal_draw(state)
        return grid, np.full(xi_points, -np.log(xi_points))
    raise ValueError("xi_rule must be one of ['gh', 'qmc', 'mc']")


def _xi_grid(q_xi: int, latent_dim: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = _gh(q_xi)
    # Match the Rust ordering: axis k advances every q_xi^k nodes.
    n_points = q_xi**latent_dim
    if n_points > 1_000_000:
        raise ValueError("q_xi ** latent_dim exceeds the tensor-grid limit; use qmc/mc")
    idx = np.arange(n_points)
    grid = np.empty((len(idx), latent_dim))
    logw = np.zeros(len(idx))
    rem = idx.copy()
    for k in range(latent_dim):
        sel = rem % q_xi
        rem //= q_xi
        grid[:, k] = nodes[sel]
        logw += np.log(weights[sel])
    return grid, logw


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0.0, -np.log1p(np.exp(-np.abs(x))), x - np.log1p(np.exp(x)))


def _build_contexts(
    pop: dict, mu: np.ndarray, sigma: np.ndarray, sigma_u: float, n_dims: int, q_u: int
) -> dict:
    kind = pop["kind"]
    if kind == "single":
        return {"n_ctx": 1, "shift": np.zeros((1, n_dims)), "scale": np.ones((1, n_dims))}
    if kind in {"multigroup", "singlefree"}:
        # singlefree (FIPC) is a one-group multigroup with free (mu, sigma)
        return {"n_ctx": mu.shape[0], "shift": mu.copy(), "scale": sigma.copy()}
    nodes, weights = _gh(q_u)
    return {
        "n_ctx": q_u,
        "shift": np.repeat((sigma_u * nodes)[:, None], n_dims, axis=1),
        "scale": np.ones((q_u, n_dims)),
        "u_nodes": nodes,
        "u_logw": np.log(weights),
    }


def _build_tables(
    alpha: np.ndarray,
    b: np.ndarray,
    zeta: np.ndarray,
    tau: float,
    model: str,
    factor_id: np.ndarray,
    ctx: dict,
    t_nodes: np.ndarray,
    x_grid: np.ndarray,
    eps_distance: float,
    n_dims: int,
    offsets: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (logp1, logp0, c0) with shapes (S, I, Qt, Nx) and (S, D, Qt, Nx)."""
    free_alpha, uses_space = _model_flags(model)
    a = np.exp(alpha) if free_alpha else np.ones_like(alpha)
    # theta value per (ctx, item, t): shift/scale of the item's dimension.
    shift = ctx["shift"][:, factor_id]  # (S, I)
    scale = ctx["scale"][:, factor_id]  # (S, I)
    theta = shift[:, :, None] + scale[:, :, None] * t_nodes[None, None, :]  # (S, I, Qt)
    eta = a[None, :, None, None] * theta[:, :, :, None] + b[None, :, None, None]
    if offsets is not None:
        eta = eta + offsets[:, :, None, None]
    kind = _interaction_kind(model)
    if kind == "distance":
        diff = x_grid[None, :, :] - zeta[:, None, :]  # (I, Nx, K)
        dist = np.sqrt(eps_distance + np.sum(diff * diff, axis=2))  # (I, Nx)
        eta = eta - np.exp(tau) * dist[None, :, None, :]
    elif kind == "inner":
        eta = eta + (zeta @ x_grid.T)[None, :, None, :]
    logp1 = _log_sigmoid(eta)
    logp0 = _log_sigmoid(-eta)
    n_ctx, n_items = eta.shape[0], eta.shape[1]
    c0 = np.zeros((n_ctx, n_dims, eta.shape[2], eta.shape[3]))
    for d in range(n_dims):
        c0[:, d] = logp0[:, factor_id == d].sum(axis=1)
    return logp1, logp0, c0


def _person_logliks(
    y: np.ndarray,
    observed: np.ndarray,
    factor_id: np.ndarray,
    logp1: np.ndarray,
    logp0: np.ndarray,
    c0: np.ndarray,
    t_logw: np.ndarray,
    x_logw: np.ndarray,
    s_of_person: np.ndarray,
    n_dims: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized person pass for one context assignment.

    Returns (l, log_zdx, log_lp): l has shape (P, D, Qt, Nx); log_zdx (P, D, Nx);
    log_lp (P,).
    """
    delta = logp1 - logp0  # (S, I, Qt, Nx)
    pos = np.where(observed, y, 0.0)  # (P, I)
    l = c0[s_of_person]  # (P, D, Qt, Nx) — copy via fancy indexing
    # positives: add delta_i; missing: subtract logp0_i — per dimension.
    for d in range(n_dims):
        items = np.flatnonzero(factor_id == d)
        # (P, I_d) @ (S,I_d,Qt,Nx) gathered per person context
        pos_d = pos[:, items]  # (P, I_d)
        miss_d = (~observed[:, items]).astype(np.float64)  # (P, I_d)
        delta_d = delta[:, items]  # (S, I_d, Qt, Nx)
        logp0_d = logp0[:, items]
        # einsum over the item axis with per-person context gather
        l[:, d] += np.einsum(
            "pi,piqx->pqx", pos_d, delta_d[s_of_person], optimize=True
        )
        if miss_d.any():
            l[:, d] -= np.einsum(
                "pi,piqx->pqx", miss_d, logp0_d[s_of_person], optimize=True
            )
    lw = t_logw[None, None, :, None] + l  # (P, D, Qt, Nx)
    m = lw.max(axis=2, keepdims=True)
    log_zdx = np.squeeze(m, axis=2) + np.log(
        np.exp(lw - m).sum(axis=2)
    )  # (P, D, Nx)
    ax = x_logw[None, :] + log_zdx.sum(axis=1)  # (P, Nx)
    mx = ax.max(axis=1, keepdims=True)
    log_lp = np.squeeze(mx, axis=1) + np.log(np.exp(ax - mx).sum(axis=1))
    return l, log_zdx, log_lp


def _posteriors(
    l: np.ndarray,
    log_zdx: np.ndarray,
    log_lp: np.ndarray,
    t_logw: np.ndarray,
    x_logw: np.ndarray,
) -> np.ndarray:
    """Joint per-person posterior over (d, t, x): shape (P, D, Qt, Nx)."""
    px = np.exp(x_logw[None, :] + log_zdx.sum(axis=1) - log_lp[:, None])  # (P, Nx)
    pt = np.exp(t_logw[None, None, :, None] + l - log_zdx[:, :, None, :])
    return px[:, None, None, :] * pt


def _accumulate(
    post: np.ndarray,
    w_outer: np.ndarray,
    y: np.ndarray,
    observed: np.ndarray,
    factor_id: np.ndarray,
    s_of_person: np.ndarray,
    n_ctx: int,
    nbar: np.ndarray,
    rbar: np.ndarray,
    mbar: np.ndarray,
) -> None:
    wpost = post * w_outer[:, None, None, None]  # (P, D, Qt, Nx)
    for s in range(n_ctx):
        sel = s_of_person == s
        if not sel.any():
            continue
        nbar[s] += wpost[sel].sum(axis=0)
        pos = np.where(observed[sel], y[sel], 0.0)  # (Ps, I)
        miss = (~observed[sel]).astype(np.float64)
        dsel = wpost[sel][:, factor_id]  # (Ps, I, Qt, Nx)
        rbar[s] += np.einsum("pi,piqx->iqx", pos, dsel, optimize=True)
        if miss.any():
            mbar[s] += np.einsum("pi,piqx->iqx", miss, dsel, optimize=True)


def _item_q(
    n_i: np.ndarray,
    r_i: np.ndarray,
    eta: np.ndarray,
    alpha_i: float,
    b_i: float,
    zeta_i: np.ndarray,
    free_alpha: bool,
    uses_space: bool,
    pen: dict,
) -> float:
    q = float(np.sum(r_i * _log_sigmoid(eta) + (n_i - r_i) * _log_sigmoid(-eta)))
    q -= 0.5 * pen["lambda_b"] * b_i * b_i
    if free_alpha:
        da = alpha_i - pen["mu_alpha"]
        q -= 0.5 * pen["lambda_alpha"] * da * da
    if uses_space:
        q -= 0.5 * pen["lambda_zeta"] * float(zeta_i @ zeta_i)
    return q


def fit_marginal_numpy(
    y: np.ndarray,
    observed: np.ndarray,
    factor_id: np.ndarray,
    model: str = "MLS2PLM",
    n_dims: int | None = None,
    latent_dim: int = 2,
    pop: dict | None = None,
    q_theta: int = 21,
    q_xi: int = 11,
    q_u: int = 15,
    max_iter: int = 200,
    tol: float = 1e-5,
    m_steps: int = 4,
    init_zeta_radius: float = 0.5,
    init_sigma_u: float = 0.3,
    eps_distance: float = 1e-8,
    penalty: dict | None = None,
    xi_rule: str = "gh",
    xi_points: int = 256,
    xi_seed: int = 0,
    anchors: dict | None = None,
    zero_inflation: bool = False,
    covariate: dict | None = None,
) -> dict:
    """NumPy mirror of ``mlsirm_core::marginal::fit_marginal``.

    ``pop`` is ``{"kind": "single"}`` (default), ``{"kind": "singlefree"}``
    (FIPC: free population mean/sd, requires ``anchors``),
    ``{"kind": "multigroup", "group_id": ..., "n_groups": ...}`` or
    ``{"kind": "multilevel", "cluster_id": ..., "n_clusters": ...}``.
    ``anchors`` is ``{"fixed": bool[I], "alpha": ..., "b": ..., "zeta": ...,
    "tau": float | None}`` — fixed items stay frozen (FIPC, Kim 2006).
    """
    y = np.asarray(y, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    factor_id = np.asarray(factor_id, dtype=np.int64)
    n_persons, n_items = y.shape
    if n_dims is None:
        n_dims = int(factor_id.max()) + 1
    model = model.upper()
    free_alpha, uses_space = _model_flags(model)
    pop = pop or {"kind": "single"}
    pen = dict(LSIRM_PRIOR)
    if penalty:
        pen.update(penalty)

    if model in {"ULS2PLM", "ULSRM"} and n_dims != 1:
        raise ValueError("unidimensional models require n_dims == 1")
    if factor_id.min() < 0 or factor_id.max() >= n_dims:
        raise ValueError("factor_id values must be in 0..n_dims-1")
    if latent_dim < 1 or latent_dim > 3:
        raise ValueError("marginal estimator supports 1 <= latent_dim <= 3")
    obs_vals = y[observed]
    if obs_vals.size and not np.all((obs_vals == 0.0) | (obs_vals == 1.0)):
        raise ValueError("observed responses must be 0 or 1")

    t_nodes, t_weights = _gh(q_theta)
    t_logw = np.log(t_weights)
    if uses_space:
        x_grid, x_logw = _xi_nodes(xi_rule, latent_dim, q_xi, xi_points, xi_seed)
    else:
        x_grid, x_logw = np.zeros((1, latent_dim)), np.zeros(1)
    n_x = len(x_logw)

    # --- deterministic init (mirror of the Rust code) ---
    counts = observed.sum(axis=0)
    means = np.where(counts > 0, np.where(observed, y, 0.0).sum(axis=0) / np.maximum(counts, 1), 0.5)
    prop = np.clip(means, 0.02, 0.98)
    b = np.log(prop / (1.0 - prop))
    alpha = np.zeros(n_items)
    zeta = np.zeros((n_items, latent_dim))
    if uses_space and _interaction_kind(model) == "inner":
        # positive-manifold start for loadings (mirror of the Rust init)
        zeta[:] = init_zeta_radius
    elif uses_space:
        angle = 2.0 * np.pi * np.arange(n_items) / max(n_items, 1)
        zeta[:, 0] = init_zeta_radius * np.cos(angle)
        if latent_dim >= 2:
            zeta[:, 1] = init_zeta_radius * np.sin(angle)
        if latent_dim >= 3:
            zeta[:, 2] = init_zeta_radius * np.cos(2.0 * angle) * 0.5
    tau = 0.0 if _interaction_kind(model) == "distance" else -30.0

    kind = pop["kind"]
    if kind == "singlefree" and anchors is None:
        raise ValueError("singlefree (FIPC) requires anchors for identification")
    if covariate is not None:
        if kind == "multilevel":
            raise ValueError("item covariates with a multilevel structure are not supported")
        n_ctx_expected = pop.get("n_groups", 1) if kind == "multigroup" else 1
        w_cov = np.asarray(covariate["w"], dtype=np.float64).reshape(
            n_ctx_expected, n_items
        )
        if n_ctx_expected == 1 and anchors is None:
            raise ValueError(
                "a single-context item covariate is collinear with b; use multigroup "
                "contexts (booklets) or anchors"
            )
    else:
        w_cov = None
    n_groups = (
        pop.get("n_groups", 0) if kind == "multigroup" else (1 if kind == "singlefree" else 0)
    )
    n_clusters = pop.get("n_clusters", 0) if kind == "multilevel" else 0
    if kind == "multigroup":
        group_id = np.asarray(pop["group_id"], dtype=np.int64)
        if group_id.shape != (n_persons,) or group_id.min() < 0 or group_id.max() >= n_groups:
            raise ValueError("group_id values must be in 0..n_groups-1")
    if kind == "multilevel":
        cluster_id = np.asarray(pop["cluster_id"], dtype=np.int64)
        if (
            cluster_id.shape != (n_persons,)
            or cluster_id.min() < 0
            or cluster_id.max() >= n_clusters
        ):
            raise ValueError("cluster_id values must be in 0..n_clusters-1")
    mu = np.zeros((n_groups, n_dims))
    sigma = np.ones((n_groups, n_dims))
    sigma_u = init_sigma_u if n_clusters else 0.0
    delta = float(covariate.get("init_delta", 0.0)) if covariate is not None else 0.0
    all_zero = ~(np.where(observed, y, 0.0) > 0).any(axis=1)
    if zero_inflation:
        frac = float(all_zero.mean())
        pi_zero = float(np.clip(0.5 * frac, 1e-4, 0.98))
    else:
        pi_zero = 0.0
    zero_resp = np.zeros(n_persons)
    fixed_mask = np.zeros(n_items, dtype=bool)
    anchor_tau = None
    if anchors is not None:
        fixed_mask = np.asarray(anchors["fixed"], dtype=bool)
        if fixed_mask.shape != (n_items,) or not fixed_mask.any():
            raise ValueError("anchors must fix at least one item and match n_items")
        alpha[fixed_mask] = np.asarray(anchors["alpha"], dtype=float)[fixed_mask]
        b[fixed_mask] = np.asarray(anchors["b"], dtype=float)[fixed_mask]
        zeta[fixed_mask] = np.asarray(anchors["zeta"], dtype=float).reshape(
            n_items, latent_dim
        )[fixed_mask]
        anchor_tau = anchors.get("tau")
        if anchor_tau is not None:
            tau = float(anchor_tau)

    loglik_trace: list[float] = []
    converged = False

    def _zi_mix(lp_irt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # mixture log-marginal and IRT-class weight, elementwise over persons
        log_pi = np.log(pi_zero) if pi_zero > 0 else -np.inf
        log_1m = np.log1p(-pi_zero)
        a_z = np.where(all_zero_bcast, log_pi, -np.inf)
        b_z = log_1m + lp_irt
        m = np.maximum(a_z, b_z)
        lp_mix = m + np.log(np.exp(a_z - m) + np.exp(b_z - m))
        return lp_mix, np.exp(b_z - lp_mix)

    for _iteration in range(max_iter):
        ctx = _build_contexts(pop, mu, sigma, sigma_u, n_dims, q_u)
        offsets = delta * w_cov if w_cov is not None else None
        logp1, logp0, c0 = _build_tables(
            alpha, b, zeta, tau, model, factor_id, ctx, t_nodes, x_grid, eps_distance,
            n_dims, offsets,
        )
        n_ctx = ctx["n_ctx"]
        nbar = np.zeros((n_ctx, n_dims, q_theta, n_x))
        rbar = np.zeros((n_ctx, n_items, q_theta, n_x))
        mbar = np.zeros((n_ctx, n_items, q_theta, n_x))

        if kind in {"single", "singlefree", "multigroup"}:
            s_of_person = (
                group_id if kind == "multigroup" else np.zeros(n_persons, dtype=np.int64)
            )
            l, log_zdx, log_lp = _person_logliks(
                y, observed, factor_id, logp1, logp0, c0, t_logw, x_logw, s_of_person, n_dims
            )
            if zero_inflation:
                all_zero_bcast = all_zero
                lp_mix, w_irt = _zi_mix(log_lp)
                loglik = float(lp_mix.sum())
                zero_resp = 1.0 - w_irt
            else:
                loglik = float(log_lp.sum())
                w_irt = np.ones(n_persons)
            post = _posteriors(l, log_zdx, log_lp, t_logw, x_logw)
            _accumulate(
                post, w_irt, y, observed, factor_id, s_of_person, n_ctx,
                nbar, rbar, mbar,
            )
            sum_e_v2 = 0.0
        else:  # multilevel
            lp_v = np.empty((n_persons, n_ctx))
            for v in range(n_ctx):
                s_all = np.full(n_persons, v, dtype=np.int64)
                _, _, lp = _person_logliks(
                    y, observed, factor_id, logp1, logp0, c0, t_logw, x_logw, s_all, n_dims
                )
                lp_v[:, v] = lp
            if zero_inflation:
                all_zero_bcast = all_zero[:, None]
                lp_v, w_irt_v = _zi_mix(lp_v)
            else:
                w_irt_v = np.ones_like(lp_v)
            log_cluster = np.zeros((n_clusters, n_ctx)) + ctx["u_logw"][None, :]
            np.add.at(log_cluster, cluster_id, lp_v)
            mc = log_cluster.max(axis=1, keepdims=True)
            lse = np.squeeze(mc, axis=1) + np.log(np.exp(log_cluster - mc).sum(axis=1))
            loglik = float(lse.sum())
            cluster_post = np.exp(log_cluster - lse[:, None])  # (C, V)
            sum_e_v2 = float((cluster_post * ctx["u_nodes"][None, :] ** 2).sum())
            if zero_inflation:
                zero_resp = (cluster_post[cluster_id] * (1.0 - w_irt_v)).sum(axis=1)
            for v in range(n_ctx):
                w_outer = cluster_post[cluster_id, v] * w_irt_v[:, v]
                keep = w_outer >= 1e-14
                if not keep.any():
                    continue
                s_all = np.full(n_persons, v, dtype=np.int64)
                l, log_zdx, log_lp = _person_logliks(
                    y, observed, factor_id, logp1, logp0, c0, t_logw, x_logw, s_all, n_dims
                )
                post = _posteriors(l, log_zdx, log_lp, t_logw, x_logw)
                w_eff = np.where(keep, w_outer, 0.0)
                _accumulate(
                    post, w_eff, y, observed, factor_id, s_all, n_ctx, nbar, rbar, mbar
                )
        loglik_trace.append(loglik)
        if zero_inflation:
            pi_zero = float(np.clip(zero_resp.mean(), 0.0, 0.999))

        # --- M-step: items (Fisher-preconditioned ascent with Armijo) ---
        gamma = float(np.exp(tau))
        theta_sx = ctx["shift"][:, :, None] + ctx["scale"][:, :, None] * t_nodes[None, None, :]
        for i in range(n_items):
            if fixed_mask[i]:
                continue
            d = int(factor_id[i])
            zeta_i = zeta[i].copy()
            n_i = nbar[:, d] - mbar[:, i]  # (S, Qt, Nx)
            r_i = rbar[:, i]
            theta_i = theta_sx[:, d]  # (S, Qt)

            off_i = (
                offsets[:, i][:, None, None] if offsets is not None else 0.0
            )

            kind_i = _interaction_kind(model)

            def eta_of(alpha_c: float, b_c: float, zeta_c: np.ndarray) -> np.ndarray:
                a_c = np.exp(alpha_c) if free_alpha else 1.0
                e = a_c * theta_i[:, :, None] + b_c + off_i
                if kind_i == "distance":
                    diff = x_grid - zeta_c[None, :]
                    dist = np.sqrt(eps_distance + np.sum(diff * diff, axis=1))
                    e = e - gamma * dist[None, None, :]
                elif kind_i == "inner":
                    e = e + (x_grid @ zeta_c)[None, None, :]
                return e

            cur_q = _item_q(
                n_i, r_i, eta_of(alpha[i], b[i], zeta_i), alpha[i], b[i], zeta_i,
                free_alpha, uses_space, pen,
            )
            for _ in range(m_steps):
                a_c = np.exp(alpha[i]) if free_alpha else 1.0
                eta = eta_of(alpha[i], b[i], zeta_i)
                prob = 1.0 / (1.0 + np.exp(-np.clip(eta, -700, 700)))
                resid = r_i - n_i * prob
                info = np.maximum(n_i * prob * (1.0 - prob), 0.0)
                g_b = float(resid.sum()) - pen["lambda_b"] * b[i]
                i_b = float(info.sum())
                if free_alpha:
                    deta_a = a_c * theta_i[:, :, None]
                    g_alpha = float((resid * deta_a).sum()) - pen["lambda_alpha"] * (
                        alpha[i] - pen["mu_alpha"]
                    )
                    i_alpha = float((info * deta_a * deta_a).sum())
                else:
                    g_alpha, i_alpha = 0.0, 0.0
                if uses_space:
                    if kind_i == "inner":
                        deta_z = x_grid  # (Nx, K)
                    else:
                        diff = x_grid - zeta_i[None, :]
                        dist = np.sqrt(eps_distance + np.sum(diff * diff, axis=1))
                        deta_z = gamma * diff / dist[:, None]  # (Nx, K)
                    g_zeta = (
                        np.einsum("stx,xk->k", resid, deta_z, optimize=True)
                        - pen["lambda_zeta"] * zeta_i
                    )
                    i_zeta = np.einsum("stx,xk->k", info, deta_z * deta_z, optimize=True)
                else:
                    g_zeta = np.zeros(latent_dim)
                    i_zeta = np.zeros(latent_dim)
                d_b = g_b / (i_b + pen["lambda_b"] + 1e-8)
                d_alpha = g_alpha / (i_alpha + pen["lambda_alpha"] + 1e-8)
                d_zeta = g_zeta / (i_zeta + pen["lambda_zeta"] + 1e-8)
                slope = g_b * d_b + g_alpha * d_alpha + float(g_zeta @ d_zeta)
                if slope < 1e-20:
                    break
                step, accepted = 1.0, False
                for _ls in range(30):
                    cand_b = b[i] + step * d_b
                    cand_alpha = alpha[i] + step * d_alpha if free_alpha else alpha[i]
                    cand_zeta = zeta_i + step * d_zeta
                    cand_q = _item_q(
                        n_i, r_i, eta_of(cand_alpha, cand_b, cand_zeta), cand_alpha,
                        cand_b, cand_zeta, free_alpha, uses_space, pen,
                    )
                    if cand_q > cur_q + 1e-4 * step * slope:
                        b[i] = cand_b
                        if free_alpha:
                            alpha[i] = float(np.clip(cand_alpha, -6.0, 3.0))
                        zeta_i = cand_zeta
                        cur_q = cand_q
                        accepted = True
                        break
                    step *= 0.5
                if not accepted:
                    break
            zeta[i] = zeta_i

        # --- M-step: tau (distance kind only) ---
        if uses_space and anchor_tau is None and _interaction_kind(model) == "distance":
            gamma = float(np.exp(tau))
            diff = x_grid[None, :, :] - zeta[:, None, :]
            dist = np.sqrt(eps_distance + np.sum(diff * diff, axis=2))  # (I, Nx)
            a_all = np.exp(alpha) if free_alpha else np.ones(n_items)
            theta_it = theta_sx[:, factor_id]  # (S, I, Qt)
            n_all = nbar[:, factor_id] - mbar  # (S, I, Qt, Nx)
            off_all = offsets[:, :, None, None] if offsets is not None else 0.0
            eta = (
                a_all[None, :, None, None] * theta_it[:, :, :, None]
                + b[None, :, None, None]
                + off_all
                - gamma * dist[None, :, None, :]
            )
            prob = 1.0 / (1.0 + np.exp(-np.clip(eta, -700, 700)))
            resid = rbar - n_all * prob
            deta = -gamma * dist[None, :, None, :]
            grad = float((resid * deta).sum()) - pen["lambda_tau"] * (tau - pen["mu_tau"])
            info = float((n_all * prob * (1.0 - prob) * deta * deta).sum()) + pen["lambda_tau"]
            if info > 0.0:
                direction = grad / info

                def total_q(tau_c: float) -> float:
                    e = (
                        a_all[None, :, None, None] * theta_it[:, :, :, None]
                        + b[None, :, None, None]
                        + off_all
                        - np.exp(tau_c) * dist[None, :, None, :]
                    )
                    qv = float(
                        np.sum(rbar * _log_sigmoid(e) + (n_all - rbar) * _log_sigmoid(-e))
                    )
                    qv -= 0.5 * pen["lambda_b"] * float(b @ b)
                    if free_alpha:
                        da = alpha - pen["mu_alpha"]
                        qv -= 0.5 * pen["lambda_alpha"] * float(da @ da)
                    qv -= 0.5 * pen["lambda_zeta"] * float((zeta * zeta).sum())
                    dt = tau_c - pen["mu_tau"]
                    return qv - 0.5 * pen["lambda_tau"] * dt * dt

                cur = total_q(tau)
                step = 1.0
                for _ls in range(20):
                    cand = float(np.clip(tau + step * direction, -10.0, 5.0))
                    if total_q(cand) > cur:
                        tau = cand
                        break
                    step *= 0.5

        # --- M-step: covariate coefficient delta (Debeer-Janssen) ---
        if w_cov is not None:
            gamma = float(np.exp(tau))
            a_all = np.exp(alpha) if free_alpha else np.ones(n_items)
            theta_it = theta_sx[:, factor_id]  # (S, I, Qt)
            n_all = nbar[:, factor_id] - mbar
            if uses_space:
                diffz = x_grid[None, :, :] - zeta[:, None, :]
                distz = np.sqrt(eps_distance + np.sum(diffz * diffz, axis=2))  # (I, Nx)
                dterm = gamma * distz[None, :, None, :]
            else:
                dterm = 0.0

            def eta_delta(delta_c: float) -> np.ndarray:
                return (
                    a_all[None, :, None, None] * theta_it[:, :, :, None]
                    + b[None, :, None, None]
                    + (delta_c * w_cov)[:, :, None, None]
                    - dterm
                )

            eta = eta_delta(delta)
            prob = 1.0 / (1.0 + np.exp(-np.clip(eta, -700, 700)))
            resid = rbar - n_all * prob
            w_bcast = w_cov[:, :, None, None]
            grad_d = float((resid * w_bcast).sum())
            info_d = float((n_all * prob * (1.0 - prob) * w_bcast * w_bcast).sum())
            if info_d > 0.0:
                direction = grad_d / info_d

                def q_of_delta(delta_c: float) -> float:
                    e = eta_delta(delta_c)
                    return float(
                        np.sum(rbar * _log_sigmoid(e) + (n_all - rbar) * _log_sigmoid(-e))
                    )

                cur = q_of_delta(delta)
                step = 1.0
                for _ls in range(20):
                    cand = float(np.clip(delta + step * direction, -10.0, 10.0))
                    if q_of_delta(cand) > cur:
                        delta = cand
                        break
                    step *= 0.5

        # --- M-step: population parameters ---
        if kind in {"multigroup", "singlefree"}:
            g_start = 0 if kind == "singlefree" else 1
            for g in range(g_start, n_groups):
                for d in range(n_dims):
                    theta_g = mu[g, d] + sigma[g, d] * t_nodes  # (Qt,)
                    w = nbar[g, d]  # (Qt, Nx)
                    w_sum = float(w.sum())
                    if w_sum > 1e-10:
                        m1 = float((w * theta_g[:, None]).sum())
                        m2 = float((w * (theta_g**2)[:, None]).sum())
                        mean = m1 / w_sum
                        var = max(m2 / w_sum - mean * mean, 0.01)
                        mu[g, d] = mean
                        sigma[g, d] = float(np.clip(np.sqrt(var), 0.1, 10.0))
        elif kind == "multilevel" and n_clusters:
            e_v2 = sum_e_v2 / n_clusters
            sigma_u = float(np.clip(np.sqrt(sigma_u * sigma_u * e_v2), 0.0, 10.0))

        if len(loglik_trace) > 1 and abs(loglik_trace[-1] - loglik_trace[-2]) < tol:
            converged = True
            break

    # --- final EAP pass ---
    ctx = _build_contexts(pop, mu, sigma, sigma_u, n_dims, q_u)
    final_offsets = delta * w_cov if w_cov is not None else None
    logp1, logp0, c0 = _build_tables(
        alpha, b, zeta, tau, model, factor_id, ctx, t_nodes, x_grid, eps_distance,
        n_dims, final_offsets,
    )
    theta_eap = np.zeros((n_persons, n_dims))
    theta_m2 = np.zeros((n_persons, n_dims))
    xi_eap = np.zeros((n_persons, latent_dim))
    u_eap = np.zeros(n_clusters)

    def eap_accumulate(s_all: np.ndarray, w_outer: np.ndarray) -> None:
        l, log_zdx, log_lp = _person_logliks(
            y, observed, factor_id, logp1, logp0, c0, t_logw, x_logw, s_all, n_dims
        )
        post = _posteriors(l, log_zdx, log_lp, t_logw, x_logw)
        wpost = post * w_outer[:, None, None, None]
        px = wpost.sum(axis=(1, 2)) / n_dims  # (P, Nx) — same for every d
        xi_eap[:] += px @ x_grid
        theta_s = ctx["shift"][s_all][:, :, None] + ctx["scale"][s_all][:, :, None] * t_nodes
        theta_eap[:] += np.einsum("pdtx,pdt->pd", wpost, theta_s, optimize=True)
        theta_m2[:] += np.einsum("pdtx,pdt->pd", wpost, theta_s**2, optimize=True)

    if kind in {"single", "singlefree"}:
        eap_accumulate(np.zeros(n_persons, dtype=np.int64), np.ones(n_persons))
    elif kind == "multigroup":
        eap_accumulate(group_id, np.ones(n_persons))
    else:
        lp_v = np.empty((n_persons, ctx["n_ctx"]))
        for v in range(ctx["n_ctx"]):
            s_all = np.full(n_persons, v, dtype=np.int64)
            _, _, lp = _person_logliks(
                y, observed, factor_id, logp1, logp0, c0, t_logw, x_logw, s_all, n_dims
            )
            lp_v[:, v] = lp
        log_cluster = np.zeros((n_clusters, ctx["n_ctx"])) + ctx["u_logw"][None, :]
        np.add.at(log_cluster, cluster_id, lp_v)
        mc = log_cluster.max(axis=1, keepdims=True)
        lse = np.squeeze(mc, axis=1) + np.log(np.exp(log_cluster - mc).sum(axis=1))
        cluster_post = np.exp(log_cluster - lse[:, None])
        u_eap[:] = cluster_post @ (sigma_u * ctx["u_nodes"])
        for v in range(ctx["n_ctx"]):
            w_outer = cluster_post[cluster_id, v]
            w_outer = np.where(w_outer >= 1e-14, w_outer, 0.0)
            if not w_outer.any():
                continue
            eap_accumulate(np.full(n_persons, v, dtype=np.int64), w_outer)

    theta_sd = np.sqrt(np.maximum(theta_m2 - theta_eap**2, 0.0))

    # free parameters: items (respecting anchors) + tau + population
    per_item = 1 + int(free_alpha) + (latent_dim if uses_space else 0)
    n_free_items = int((~fixed_mask).sum())
    tau_free = (
        uses_space and anchor_tau is None and _interaction_kind(model) == "distance"
    )
    pop_params = {
        "single": 0,
        "singlefree": 2 * n_dims,
        "multigroup": 2 * n_dims * max(n_groups - 1, 0),
        "multilevel": 1,
    }[kind]
    n_parameters = (
        n_free_items * per_item
        + int(tau_free)
        + pop_params
        + int(zero_inflation)
        + int(w_cov is not None)
    )
    ll_final = loglik_trace[-1] if loglik_trace else float("nan")
    k, nf = float(n_parameters), float(n_persons)
    dev = -2.0 * ll_final
    aic = dev + 2.0 * k
    ic = {
        "aic": aic,
        "bic": dev + k * np.log(nf),
        "aicc": aic + 2.0 * k * (k + 1.0) / (nf - k - 1.0) if nf - k - 1.0 > 0 else float("nan"),
        "sabic": dev + k * np.log((nf + 2.0) / 24.0),
        "caic": dev + k * (np.log(nf) + 1.0),
        "n_parameters": n_parameters,
        "n": n_persons,
    }

    if uses_space and anchors is None:
        # anchored calibrations inherit the anchor orientation
        _pca_align(zeta, xi_eap)

    return {
        "alpha": alpha,
        "b": b,
        "zeta": zeta,
        "tau": float(tau),
        "theta_eap": theta_eap,
        "theta_sd": theta_sd,
        "xi_eap": xi_eap,
        "mu": mu,
        "sigma": sigma,
        "sigma_u": float(sigma_u),
        "u_eap": u_eap,
        "loglik_trace": loglik_trace,
        "n_iter": len(loglik_trace),
        "converged": converged,
        "status": "converged" if converged else "max_iter_reached",
        "ic": ic,
        "delta": float(delta),
        "pi_zero": float(pi_zero),
        "zero_responsibility": zero_resp if zero_inflation else np.zeros(0),
    }


def _pca_align(zeta: np.ndarray, xi: np.ndarray) -> None:
    """In-place rotation/reflection alignment (mirror of the Rust Jacobi code):
    principal axes of the uncentered second moment of ``zeta``, columns ordered
    by descending eigenvalue, sign fixed by the largest-|coordinate| item."""
    k = zeta.shape[1]
    if k == 1:
        i = int(np.argmax(np.abs(zeta[:, 0])))
        if zeta[i, 0] < 0.0:
            zeta *= -1.0
            xi *= -1.0
        return
    m = zeta.T @ zeta
    evals, evecs = np.linalg.eigh(m)
    order = np.argsort(evals)[::-1]
    rot = evecs[:, order]
    zeta[:] = zeta @ rot
    xi[:] = xi @ rot
    for c in range(k):
        i = int(np.argmax(np.abs(zeta[:, c])))
        if zeta[i, c] < 0.0:
            zeta[:, c] *= -1.0
            xi[:, c] *= -1.0


def score_eap(
    y: np.ndarray,
    observed: np.ndarray,
    factor_id: np.ndarray,
    alpha: np.ndarray,
    b: np.ndarray,
    zeta: np.ndarray,
    tau: float,
    model: str = "MLS2PLM",
    n_dims: int | None = None,
    q_theta: int = 21,
    q_xi: int = 11,
    eps_distance: float = 1e-8,
) -> dict:
    """EAP scoring of response vectors with **frozen** item parameters.

    The serving-side counterpart of :func:`fit_marginal_numpy`: one E-step
    pass under the standard `N(0, 1)` population prior, no parameter updates.
    Returns per-person `theta_eap`, `theta_sd`, `xi_eap`, and the marginal
    log-likelihood of each response vector.
    """
    y = np.asarray(y, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    factor_id = np.asarray(factor_id, dtype=np.int64)
    if n_dims is None:
        n_dims = int(factor_id.max()) + 1
    model = model.upper()
    _, uses_space = _model_flags(model)
    alpha = np.asarray(alpha, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    zeta = np.asarray(zeta, dtype=np.float64)
    latent_dim = zeta.shape[1]
    n_persons = y.shape[0]

    t_nodes, t_weights = _gh(q_theta)
    t_logw = np.log(t_weights)
    if uses_space:
        x_grid, x_logw = _xi_grid(q_xi, latent_dim)
    else:
        x_grid, x_logw = np.zeros((1, latent_dim)), np.zeros(1)

    ctx = {"n_ctx": 1, "shift": np.zeros((1, n_dims)), "scale": np.ones((1, n_dims))}
    logp1, logp0, c0 = _build_tables(
        alpha, b, zeta, float(tau), model, factor_id, ctx, t_nodes, x_grid,
        eps_distance, n_dims,
    )
    s_all = np.zeros(n_persons, dtype=np.int64)
    y_filled = np.where(observed, y, 0.0)
    l, log_zdx, log_lp = _person_logliks(
        y_filled, observed, factor_id, logp1, logp0, c0, t_logw, x_logw, s_all, n_dims
    )
    post = _posteriors(l, log_zdx, log_lp, t_logw, x_logw)
    px = post.sum(axis=(1, 2)) / n_dims  # (P, Nx)
    xi_eap = px @ x_grid
    theta_eap = np.einsum("pdtx,t->pd", post, t_nodes, optimize=True)
    theta_m2 = np.einsum("pdtx,t->pd", post, t_nodes**2, optimize=True)
    theta_sd = np.sqrt(np.maximum(theta_m2 - theta_eap**2, 0.0))
    return {
        "theta_eap": theta_eap,
        "theta_sd": theta_sd,
        "xi_eap": xi_eap,
        "loglik": log_lp,
    }
