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


def _xi_grid(q_xi: int, latent_dim: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = _gh(q_xi)
    # Match the Rust ordering: axis k advances every q_xi^k nodes.
    idx = np.arange(q_xi**latent_dim)
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
    if kind == "multigroup":
        return {"n_ctx": pop["n_groups"], "shift": mu.copy(), "scale": sigma.copy()}
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (logp1, logp0, c0) with shapes (S, I, Qt, Nx) and (S, D, Qt, Nx)."""
    free_alpha, uses_space = _model_flags(model)
    a = np.exp(alpha) if free_alpha else np.ones_like(alpha)
    # theta value per (ctx, item, t): shift/scale of the item's dimension.
    shift = ctx["shift"][:, factor_id]  # (S, I)
    scale = ctx["scale"][:, factor_id]  # (S, I)
    theta = shift[:, :, None] + scale[:, :, None] * t_nodes[None, None, :]  # (S, I, Qt)
    eta = a[None, :, None, None] * theta[:, :, :, None] + b[None, :, None, None]
    if uses_space:
        diff = x_grid[None, :, :] - zeta[:, None, :]  # (I, Nx, K)
        dist = np.sqrt(eps_distance + np.sum(diff * diff, axis=2))  # (I, Nx)
        eta = eta - np.exp(tau) * dist[None, :, None, :]
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
) -> dict:
    """NumPy mirror of ``mlsirm_core::marginal::fit_marginal``.

    ``pop`` is ``{"kind": "single"}`` (default),
    ``{"kind": "multigroup", "group_id": ..., "n_groups": ...}`` or
    ``{"kind": "multilevel", "cluster_id": ..., "n_clusters": ...}``.
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
        x_grid, x_logw = _xi_grid(q_xi, latent_dim)
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
    if uses_space:
        angle = 2.0 * np.pi * np.arange(n_items) / max(n_items, 1)
        zeta[:, 0] = init_zeta_radius * np.cos(angle)
        if latent_dim >= 2:
            zeta[:, 1] = init_zeta_radius * np.sin(angle)
        if latent_dim >= 3:
            zeta[:, 2] = init_zeta_radius * np.cos(2.0 * angle) * 0.5
    tau = 0.0 if uses_space else -30.0

    kind = pop["kind"]
    n_groups = pop.get("n_groups", 0) if kind == "multigroup" else 0
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

    loglik_trace: list[float] = []
    converged = False

    for _iteration in range(max_iter):
        ctx = _build_contexts(pop, mu, sigma, sigma_u, n_dims, q_u)
        logp1, logp0, c0 = _build_tables(
            alpha, b, zeta, tau, model, factor_id, ctx, t_nodes, x_grid, eps_distance, n_dims
        )
        n_ctx = ctx["n_ctx"]
        nbar = np.zeros((n_ctx, n_dims, q_theta, n_x))
        rbar = np.zeros((n_ctx, n_items, q_theta, n_x))
        mbar = np.zeros((n_ctx, n_items, q_theta, n_x))

        if kind == "single":
            s_of_person = np.zeros(n_persons, dtype=np.int64)
            l, log_zdx, log_lp = _person_logliks(
                y, observed, factor_id, logp1, logp0, c0, t_logw, x_logw, s_of_person, n_dims
            )
            loglik = float(log_lp.sum())
            post = _posteriors(l, log_zdx, log_lp, t_logw, x_logw)
            _accumulate(
                post, np.ones(n_persons), y, observed, factor_id, s_of_person, n_ctx,
                nbar, rbar, mbar,
            )
            sum_e_v2 = 0.0
        elif kind == "multigroup":
            s_of_person = group_id
            l, log_zdx, log_lp = _person_logliks(
                y, observed, factor_id, logp1, logp0, c0, t_logw, x_logw, s_of_person, n_dims
            )
            loglik = float(log_lp.sum())
            post = _posteriors(l, log_zdx, log_lp, t_logw, x_logw)
            _accumulate(
                post, np.ones(n_persons), y, observed, factor_id, s_of_person, n_ctx,
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
            log_cluster = np.zeros((n_clusters, n_ctx)) + ctx["u_logw"][None, :]
            np.add.at(log_cluster, cluster_id, lp_v)
            mc = log_cluster.max(axis=1, keepdims=True)
            lse = np.squeeze(mc, axis=1) + np.log(np.exp(log_cluster - mc).sum(axis=1))
            loglik = float(lse.sum())
            cluster_post = np.exp(log_cluster - lse[:, None])  # (C, V)
            sum_e_v2 = float((cluster_post * ctx["u_nodes"][None, :] ** 2).sum())
            for v in range(n_ctx):
                w_outer = cluster_post[cluster_id, v]
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

        # --- M-step: items (Fisher-preconditioned ascent with Armijo) ---
        gamma = float(np.exp(tau))
        theta_sx = ctx["shift"][:, :, None] + ctx["scale"][:, :, None] * t_nodes[None, None, :]
        for i in range(n_items):
            d = int(factor_id[i])
            zeta_i = zeta[i].copy()
            n_i = nbar[:, d] - mbar[:, i]  # (S, Qt, Nx)
            r_i = rbar[:, i]
            theta_i = theta_sx[:, d]  # (S, Qt)

            def eta_of(alpha_c: float, b_c: float, zeta_c: np.ndarray) -> np.ndarray:
                a_c = np.exp(alpha_c) if free_alpha else 1.0
                e = a_c * theta_i[:, :, None] + b_c
                if uses_space:
                    diff = x_grid - zeta_c[None, :]
                    dist = np.sqrt(eps_distance + np.sum(diff * diff, axis=1))
                    e = e - gamma * dist[None, None, :]
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

        # --- M-step: tau ---
        if uses_space:
            gamma = float(np.exp(tau))
            diff = x_grid[None, :, :] - zeta[:, None, :]
            dist = np.sqrt(eps_distance + np.sum(diff * diff, axis=2))  # (I, Nx)
            a_all = np.exp(alpha) if free_alpha else np.ones(n_items)
            theta_it = theta_sx[:, factor_id]  # (S, I, Qt)
            n_all = nbar[:, factor_id] - mbar  # (S, I, Qt, Nx)
            eta = (
                a_all[None, :, None, None] * theta_it[:, :, :, None]
                + b[None, :, None, None]
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

        # --- M-step: population parameters ---
        if kind == "multigroup":
            for g in range(1, n_groups):
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
    logp1, logp0, c0 = _build_tables(
        alpha, b, zeta, tau, model, factor_id, ctx, t_nodes, x_grid, eps_distance, n_dims
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

    if kind == "single":
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

    if uses_space:
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
