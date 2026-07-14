"""Likelihood-based item- and person-fit statistics for marginal (MMLE) fits.

Implements, for the latent-space model family (see
``docs/papers/mmle-lsirm-formula-compilation.md`` §7-§8 for the sourced
formulas):

- **S-X²** (Orlando & Thissen, 2000): summed-score item fit with the
  Lord-Wingersky (1984) recursion, generalized to the joint (theta, xi)
  quadrature grid — the recursion runs at every grid node and the expected
  proportions marginalize over the node weights. Scores are computed within
  each trait dimension (simple structure), score groups are collapsed to a
  minimum expected count, and p-values use the chi-square upper tail.
- **l_z** (Drasgow, Levine & Williams, 1985) and **l_z*** (Snijders, 2001):
  standardized person-fit log-likelihood statistics evaluated at the EAP
  trait score with the person's latent-space position fixed at its EAP (a
  documented approximation for the interaction term), using the MAP-case
  correction ``r_0(theta) = -(theta - prior_mean)`` for the N(prior_mean, 1)
  trait prior.

No SciPy (repo constraint): the chi-square survival function is computed via
the regularized upper incomplete gamma function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .estimators.marginal import _gh, _xi_grid


def _core_module():
    """The compiled Rust core, when built — the compute path for every
    statistic here (the NumPy bodies below are the parity reference and
    fallback)."""
    try:
        from . import _core  # type: ignore

        return _core
    except Exception:  # pragma: no cover
        return None


def _bank_args(params, factor_id, model, n_dims, eps_distance):
    zeta = np.asarray(params.zeta, dtype=np.float64)
    return dict(
        alpha=np.asarray(params.alpha, dtype=np.float64),
        b=np.asarray(params.b, dtype=np.float64),
        zeta=zeta.ravel(),
        tau=float(params.tau),
        factor_id=np.asarray(factor_id, dtype=np.int64),
        model=model,
        n_dims=int(n_dims),
        latent_dim=int(zeta.shape[1]),
        eps_distance=float(eps_distance),
    )


# --------------------------------------------------------------------------
# chi-square survival function (regularized upper incomplete gamma), no SciPy
# --------------------------------------------------------------------------


def _gammainc_upper_reg(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) (Numerical Recipes 6.2)."""
    if x < 0.0 or a <= 0.0:
        raise ValueError("invalid arguments to Q(a, x)")
    if x == 0.0:
        return 1.0
    if x < a + 1.0:
        # series for P(a,x), return 1 - P
        ap = a
        total = 1.0 / a
        delta = total
        for _ in range(500):
            ap += 1.0
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * 1e-15:
                break
        p = total * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return max(0.0, min(1.0, 1.0 - p))
    # continued fraction for Q(a,x) (modified Lentz)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return max(0.0, min(1.0, h * math.exp(-x + a * math.log(x) - math.lgamma(a))))


def chi2_sf(x: float, df: float) -> float:
    """P(Chi2_df >= x)."""
    if df <= 0:
        return float("nan")
    return _gammainc_upper_reg(df / 2.0, max(x, 0.0) / 2.0)


def benjamini_hochberg(p_values: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Boolean rejection mask controlling FDR at level q (BH 1995)."""
    p = np.asarray(p_values, dtype=float)
    valid = np.isfinite(p)
    m = int(valid.sum())
    reject = np.zeros(p.shape, dtype=bool)
    if m == 0:
        return reject
    order = np.argsort(np.where(valid, p, np.inf))
    ranked = p[order][:m]
    thresh = q * (np.arange(1, m + 1) / m)
    below = ranked <= thresh
    if below.any():
        k = int(np.max(np.nonzero(below)[0]))
        reject[order[: k + 1]] = True
    return reject


# --------------------------------------------------------------------------
# marginal item response tables on the quadrature grid
# --------------------------------------------------------------------------


def _icc_grid(
    params,
    factor_id: np.ndarray,
    model: str,
    q_theta: int = 21,
    q_xi: int = 11,
    eps_distance: float = 1e-8,
    prior_mean: np.ndarray | None = None,
):
    """Item ICCs on the joint (t, x) grid.

    Returns (probs (I, Qt, Nx), node weights (Qt,), (Nx,), theta nodes (Qt,)).
    ``prior_mean`` optionally shifts the trait prior per dimension (D,) — used
    for multigroup/multilevel populations where theta_d ~ N(mean_d, 1).
    """
    model = model.upper()
    free_alpha = model not in {"MLSRM", "ULSRM"}
    uses_space = model != "MIRT"
    t_nodes, t_w = _gh(q_theta)
    if uses_space:
        x_grid, x_logw = _xi_grid(q_xi, params.zeta.shape[1])
        x_w = np.exp(x_logw)
    else:
        x_grid = np.zeros((1, params.zeta.shape[1]))
        x_w = np.ones(1)
    a = np.exp(params.alpha) if free_alpha else np.ones_like(params.alpha)
    d_of_i = np.asarray(factor_id, dtype=np.int64)
    shift = np.zeros(int(d_of_i.max()) + 1) if prior_mean is None else np.asarray(prior_mean)
    theta = shift[d_of_i][:, None] + t_nodes[None, :]  # (I, Qt)
    eta = a[:, None, None] * theta[:, :, None] + params.b[:, None, None]
    if uses_space:
        diff = x_grid[None, :, :] - params.zeta[:, None, :]
        dist = np.sqrt(eps_distance + np.sum(diff * diff, axis=2))  # (I, Nx)
        eta = eta - math.exp(params.tau) * dist[:, None, :]
    probs = 1.0 / (1.0 + np.exp(-np.clip(eta, -700, 700)))
    return probs, t_w, x_w, t_nodes


def _lord_wingersky(probs: np.ndarray) -> np.ndarray:
    """Summed-score distribution at each grid node.

    ``probs`` is (I, Q) — item success probabilities at Q nodes. Returns
    (I+1, Q): P(score = r | node q) for the item set.
    """
    n_items, n_nodes = probs.shape
    f = np.zeros((n_items + 1, n_nodes))
    f[0] = 1.0 - probs[0]
    f[1] = probs[0]
    for n in range(1, n_items):
        p = probs[n]
        prev = f[: n + 1].copy()
        f[: n + 2] = 0.0
        f[: n + 1] += prev * (1.0 - p)[None, :]
        f[1 : n + 2] += prev * p[None, :]
    return f


@dataclass
class SX2Result:
    statistic: np.ndarray
    df: np.ndarray
    p_value: np.ndarray
    flagged_bh: np.ndarray
    n_score_groups: np.ndarray
    # N_s-weighted RMS of (O - E): the practical-significance effect size that
    # keeps the over-powered chi-square honest at large N (Sinharay & Haberman
    # 2014). flagged_bh requires BH significance AND rms >= min_effect.
    rms_residual: np.ndarray | None = None


def s_x2(
    responses: np.ndarray,
    factor_id: np.ndarray,
    params,
    model: str,
    mask: np.ndarray | None = None,
    q_theta: int = 21,
    q_xi: int = 11,
    eps_distance: float = 1e-8,
    prior_mean: np.ndarray | None = None,
    min_expected: float = 1.0,
    fdr_q: float = 0.05,
    person_weight: np.ndarray | None = None,
    min_effect: float = 0.0,
) -> SX2Result:
    """Orlando-Thissen S-X² per item, summed scores within each trait dim.

    Persons with any missing response inside a dimension are excluded from
    that dimension's observed table (the summed score would not be
    comparable). ``person_weight`` (0/1) can down-weight aberrant respondents
    flagged by person fit before item decisions (design doc §6).
    ``min_effect`` guards the BH flag with the RMS observed-minus-expected
    effect size (practical significance at large N).
    """
    core = _core_module()
    if core is not None and prior_mean is None:
        y0 = np.asarray(responses, dtype=float)
        observed0 = ~np.isnan(y0) if mask is None else np.asarray(mask, dtype=bool)
        d_of_i = np.asarray(factor_id, dtype=np.int64)
        n_dims = int(d_of_i.max()) + 1
        bank = _bank_args(params, d_of_i, model, n_dims, eps_distance)
        res = core.s_x2_stat(
            np.where(observed0, y0, 0.0).ravel(),
            observed0.ravel(),
            int(y0.shape[0]),
            bank["alpha"], bank["b"], bank["zeta"], bank["tau"], bank["factor_id"],
            bank["model"], bank["n_dims"], bank["latent_dim"], bank["eps_distance"],
            np.zeros(n_dims), np.ones(n_dims),
            q_theta=int(q_theta), xi_rule="gh", q_xi=int(q_xi),
            min_expected=float(min_expected), fdr_q=float(fdr_q),
            min_effect=float(min_effect),
            person_weight=None
            if person_weight is None
            else np.asarray(person_weight, dtype=np.float64),
        )
        return SX2Result(
            statistic=np.asarray(res["statistic"]),
            df=np.asarray(res["df"]),
            p_value=np.asarray(res["p_value"]),
            flagged_bh=np.asarray(res["flagged_bh"], dtype=bool),
            n_score_groups=np.asarray(res["n_score_groups"], dtype=int),
            rms_residual=np.asarray(res["rms_residual"]),
        )
    y = np.asarray(responses, dtype=float)
    observed = ~np.isnan(y) if mask is None else np.asarray(mask, dtype=bool)
    if mask is None:
        y = np.where(observed, y, 0.0)
    n_persons, n_items = y.shape
    d_of_i = np.asarray(factor_id, dtype=np.int64)
    n_dims = int(d_of_i.max()) + 1
    weight = np.ones(n_persons) if person_weight is None else np.asarray(person_weight, float)

    probs, t_w, x_w, _ = _icc_grid(
        params, d_of_i, model, q_theta, q_xi, eps_distance, prior_mean
    )
    n_free = {"MLSRM": 1, "ULSRM": 1}.get(model.upper(), 2)
    if model.upper() != "MIRT":
        n_free += params.zeta.shape[1]

    stat = np.full(n_items, np.nan)
    dof = np.full(n_items, np.nan)
    pval = np.full(n_items, np.nan)
    rms = np.full(n_items, np.nan)
    n_groups_out = np.zeros(n_items, dtype=int)

    for d in range(n_dims):
        items = np.flatnonzero(d_of_i == d)
        if len(items) < 2:
            continue
        complete = observed[:, items].all(axis=1) & (weight > 0)
        yd = y[np.ix_(complete, items)]
        n_d = len(items)
        if yd.shape[0] == 0:
            continue
        scores = yd.sum(axis=1).astype(int)
        # grid: flatten (t, x) nodes with product weights
        p_flat = probs[items].reshape(n_d, -1)  # (I_d, Qt*Nx)
        w_flat = (t_w[:, None] * x_w[None, :]).reshape(-1)
        s_all = _lord_wingersky(p_flat)  # (I_d+1, Q)
        denom = s_all @ w_flat  # (I_d+1,)
        for local_i, i in enumerate(items):
            rest = np.delete(np.arange(n_d), local_i)
            s_rest = _lord_wingersky(p_flat[rest]) if len(rest) else None
            # E_is for s = 1..I_d-1
            e = np.full(n_d + 1, np.nan)
            for s_score in range(1, n_d):
                num = float((p_flat[local_i] * s_rest[s_score - 1]) @ w_flat)
                den = float(denom[s_score])
                e[s_score] = num / den if den > 0 else np.nan
            obs_n = np.bincount(scores, minlength=n_d + 1).astype(float)
            obs_r = np.bincount(scores, weights=yd[:, local_i], minlength=n_d + 1)
            # score groups 1..I_d-1; collapse adjacent until expected >= min_expected
            groups: list[tuple[float, float, float]] = []  # (N, O_sum, E_sum)
            acc_n, acc_r, acc_e = 0.0, 0.0, 0.0
            for s_score in range(1, n_d):
                if not np.isfinite(e[s_score]):
                    continue
                acc_n += obs_n[s_score]
                acc_r += obs_r[s_score]
                acc_e += obs_n[s_score] * e[s_score]
                if acc_n > 0 and acc_e >= min_expected and (acc_n - acc_e) >= min_expected:
                    groups.append((acc_n, acc_r, acc_e))
                    acc_n, acc_r, acc_e = 0.0, 0.0, 0.0
            if acc_n > 0 and groups:
                n0, r0, e0 = groups[-1]
                groups[-1] = (n0 + acc_n, r0 + acc_r, e0 + acc_e)
            elif acc_n > 0:
                groups.append((acc_n, acc_r, acc_e))
            x2, n_grp = 0.0, 0
            rss, n_tot = 0.0, 0.0
            for gn, gr, ge in groups:
                if gn <= 0:
                    continue
                e_prop = ge / gn
                if e_prop <= 0.0 or e_prop >= 1.0:
                    continue
                o_prop = gr / gn
                x2 += gn * (o_prop - e_prop) ** 2 / (e_prop * (1.0 - e_prop))
                rss += gn * (o_prop - e_prop) ** 2
                n_tot += gn
                n_grp += 1
            df_i = n_grp - n_free
            stat[i] = x2
            n_groups_out[i] = n_grp
            rms[i] = np.sqrt(rss / n_tot) if n_tot > 0 else np.nan
            if df_i >= 1:
                dof[i] = df_i
                pval[i] = chi2_sf(x2, df_i)
    flagged = benjamini_hochberg(pval, fdr_q)
    flagged &= np.where(np.isfinite(rms), rms, -np.inf) >= min_effect
    return SX2Result(
        statistic=stat,
        df=dof,
        p_value=pval,
        flagged_bh=flagged,
        n_score_groups=n_groups_out,
        rms_residual=rms,
    )


# --------------------------------------------------------------------------
# person fit: l_z and Snijders l_z*
# --------------------------------------------------------------------------


@dataclass
class PersonFitResult:
    lz: np.ndarray
    lz_star: np.ndarray
    flagged: np.ndarray


def person_fit(
    responses: np.ndarray,
    factor_id: np.ndarray,
    params,
    model: str,
    mask: np.ndarray | None = None,
    eps_distance: float = 1e-8,
    prior_mean: np.ndarray | None = None,
    flag_threshold: float = -1.645,
) -> PersonFitResult:
    """l_z and l_z* per person and trait dimension, at EAP estimates.

    Returns arrays of shape (n_persons, n_dims). ``l_z*`` uses Snijders'
    (2001) correction with the MAP ``r_0 = -(theta_hat - prior_mean)`` term
    for the N(prior_mean, 1) trait prior (EAP ≈ MAP for these posteriors);
    the latent-space position is held at its EAP, so the correction covers
    the trait estimate only (documented approximation).
    """
    model = model.upper()
    free_alpha = model not in {"MLSRM", "ULSRM"}
    uses_space = model != "MIRT"
    y = np.asarray(responses, dtype=float)
    observed = ~np.isnan(y) if mask is None else np.asarray(mask, dtype=bool)
    y = np.where(observed, y, 0.0)
    n_persons, n_items = y.shape
    d_of_i = np.asarray(factor_id, dtype=np.int64)
    n_dims = int(d_of_i.max()) + 1
    core = _core_module()
    if core is not None:
        bank = _bank_args(params, d_of_i, model, n_dims, eps_distance)
        res = core.person_fit_stat(
            y.ravel(),
            observed.ravel(),
            int(n_persons),
            bank["alpha"], bank["b"], bank["zeta"], bank["tau"], bank["factor_id"],
            bank["model"], bank["n_dims"], bank["latent_dim"], bank["eps_distance"],
            np.asarray(params.theta, dtype=np.float64).ravel(),
            np.asarray(params.xi, dtype=np.float64).ravel(),
            prior_mean=None
            if prior_mean is None
            else np.broadcast_to(
                np.asarray(prior_mean, dtype=np.float64), (n_persons, n_dims)
            ).ravel().copy(),
            flag_threshold=float(flag_threshold),
        )
        return PersonFitResult(
            lz=np.asarray(res["lz"]).reshape(n_persons, n_dims),
            lz_star=np.asarray(res["lz_star"]).reshape(n_persons, n_dims),
            flagged=np.asarray(res["flagged"], dtype=bool),
        )
    theta = np.asarray(params.theta, dtype=float)
    a = np.exp(params.alpha) if free_alpha else np.ones(n_items)
    shift = np.zeros((n_persons, n_dims))
    if prior_mean is not None:
        shift += np.asarray(prior_mean)

    # eta_pi at EAP estimates
    eta = a[None, :] * theta[:, d_of_i] + params.b[None, :]
    if uses_space:
        diff = np.asarray(params.xi)[:, None, :] - np.asarray(params.zeta)[None, :, :]
        dist = np.sqrt(eps_distance + np.sum(diff * diff, axis=2))
        eta = eta - math.exp(params.tau) * dist
    p = 1.0 / (1.0 + np.exp(-np.clip(eta, -700, 700)))
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    w = np.log(p / (1.0 - p))  # w_i(theta) for l_z
    var_i = p * (1.0 - p)

    lz = np.full((n_persons, n_dims), np.nan)
    lz_star = np.full((n_persons, n_dims), np.nan)
    for d in range(n_dims):
        items = d_of_i == d
        o = observed[:, items]
        yd, pd, wd, vd = y[:, items], p[:, items], w[:, items], var_i[:, items]
        ad = a[items]
        n_obs = o.sum(axis=1)
        ok = n_obs >= 2
        # l_z
        w_stat = ((yd - pd) * wd * o).sum(axis=1)
        var_l = (vd * wd**2 * o).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            lz[:, d] = np.where(ok & (var_l > 0), w_stat / np.sqrt(var_l), np.nan)
        # l_z*: r_i = P'/(P(1-P)) = a_i ; c = sum P' w / sum P' r
        p_prime = ad[None, :] * vd  # (P, I_d)
        num_c = (p_prime * wd * o).sum(axis=1)
        den_c = (p_prime * ad[None, :] * o).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.where(den_c > 0, num_c / den_c, 0.0)
        w_tilde = wd - c[:, None] * ad[None, :]
        tau2 = (w_tilde**2 * vd * o).sum(axis=1) / np.maximum(n_obs, 1)
        r0 = -(theta[:, d] - shift[:, d])  # MAP correction, N(mean, 1) prior
        with np.errstate(divide="ignore", invalid="ignore"):
            lz_star[:, d] = np.where(
                ok & (tau2 > 0),
                (w_stat + c * r0) / (np.sqrt(np.maximum(n_obs, 1)) * np.sqrt(tau2)),
                np.nan,
            )
    flagged = np.nanmin(np.where(np.isnan(lz_star), np.inf, lz_star), axis=1) < flag_threshold
    return PersonFitResult(lz=lz, lz_star=lz_star, flagged=flagged)


# --------------------------------------------------------------------------
# infit / outfit at the marginal EAP estimates
# --------------------------------------------------------------------------


def infit_outfit(
    responses: np.ndarray,
    factor_id: np.ndarray,
    params,
    model: str,
    mask: np.ndarray | None = None,
    eps_distance: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Per-item infit/outfit mean squares at the EAP estimates."""
    model = model.upper()
    free_alpha = model not in {"MLSRM", "ULSRM"}
    uses_space = model != "MIRT"
    y = np.asarray(responses, dtype=float)
    observed = ~np.isnan(y) if mask is None else np.asarray(mask, dtype=bool)
    y = np.where(observed, y, 0.0)
    d_of_i = np.asarray(factor_id, dtype=np.int64)
    core = _core_module()
    if core is not None:
        n_persons = y.shape[0]
        n_dims = int(d_of_i.max()) + 1
        bank = _bank_args(params, d_of_i, model, n_dims, eps_distance)
        res = core.infit_outfit_stat(
            y.ravel(),
            observed.ravel(),
            int(n_persons),
            bank["alpha"], bank["b"], bank["zeta"], bank["tau"], bank["factor_id"],
            bank["model"], bank["n_dims"], bank["latent_dim"], bank["eps_distance"],
            np.asarray(params.theta, dtype=np.float64).ravel(),
            np.asarray(params.xi, dtype=np.float64).ravel(),
        )
        return {"infit": np.asarray(res["infit"]), "outfit": np.asarray(res["outfit"])}
    a = np.exp(params.alpha) if free_alpha else np.ones(len(params.b))
    eta = a[None, :] * np.asarray(params.theta)[:, d_of_i] + params.b[None, :]
    if uses_space:
        diff = np.asarray(params.xi)[:, None, :] - np.asarray(params.zeta)[None, :, :]
        dist = np.sqrt(eps_distance + np.sum(diff * diff, axis=2))
        eta = eta - math.exp(params.tau) * dist
    p = np.clip(1.0 / (1.0 + np.exp(-np.clip(eta, -700, 700))), 1e-12, 1 - 1e-12)
    v = p * (1.0 - p)
    resid2 = (y - p) ** 2 * observed
    n_obs = np.maximum(observed.sum(axis=0), 1)
    outfit = (resid2 / v * observed).sum(axis=0) / n_obs
    infit = resid2.sum(axis=0) / np.maximum((v * observed).sum(axis=0), 1e-12)
    return {"infit": infit, "outfit": outfit}


# --------------------------------------------------------------------------
# item screening pipeline (design doc §6 / formula compilation §9)
# --------------------------------------------------------------------------


@dataclass
class ItemScreeningRound:
    round_index: int
    kept_items: list[str]
    removed_items: list[str]
    reasons: dict[str, list[str]]
    flags: dict[str, dict[str, bool]] = field(default_factory=dict)


@dataclass
class ItemScreeningResult:
    kept_items: list[str]
    removed_items: dict[str, list[str]]
    rounds: list[ItemScreeningRound]
    final_result: object


def select_items(
    responses: np.ndarray,
    factor_id: np.ndarray,
    item_codes: list[str] | None = None,
    config=None,
    mask: np.ndarray | None = None,
    group_id: np.ndarray | None = None,
    cluster_id: np.ndarray | None = None,
    min_positive: int = 20,
    fdr_q: float = 0.05,
    sx2_min_effect: float = 0.02,
    msq_band: tuple[float, float] = (0.7, 1.3),
    min_discrimination: float = 0.35,
    isolation_z: float = 3.0,
    min_items_per_dim: int = 4,
    max_rounds: int = 5,
    min_flags_to_remove: int = 2,
    person_flag_threshold: float = -1.645,
) -> ItemScreeningResult:
    """Iterative fit -> flag -> remove -> refit item screening.

    Flags per round (literature-grounded; see the formula compilation §9):

    1. ``sparse``: fewer than ``min_positive`` positive (or negative)
       observed responses — removed on this flag alone (the item cannot
       support its parameters).
    2. ``sx2``: S-X² significant after Benjamini-Hochberg at ``fdr_q`` AND a
       practical effect (`rms_residual >= sx2_min_effect`) — chi-square power
       grows without bound in N, so significance alone over-prunes large
       calibrations (Sinharay & Haberman 2014).
    3. ``msq``: **infit** outside ``msq_band`` (Wright & Linacre 1994). Outfit
       is reported but does not gate: with very low pass rates a handful of
       surprising responses explodes the unweighted mean square.
    4. ``low_disc``: discrimination below ``min_discrimination`` (2PL models).
    5. ``isolated``: gamma-weighted mean distance to respondents is a robust
       z-score outlier above ``isolation_z`` — the LSIRM reading of an item
       nobody interacts with.

    An item is removed when it fails ``min_flags_to_remove`` of flags 2-5 (or
    flag 1 alone). Persons flagged by ``l_z* < -1.645`` are excluded from the
    flagging statistics (not from the final fit). Dimensions never drop below
    ``min_items_per_dim`` items — the worst offenders are retained with a
    note. The final refit uses all surviving items.
    """
    from .config import FitConfig
    from .fit import fit

    y = np.asarray(responses, dtype=float)
    observed = ~np.isnan(y) if mask is None else np.asarray(mask, dtype=bool)
    n_items = y.shape[1]
    d_of_i = np.asarray(factor_id, dtype=np.int64)
    codes = item_codes or [f"item_{i:03d}" for i in range(n_items)]
    config = config or FitConfig(model="MLS2PLM", estimator="mmle")
    if config.estimator != "mmle":
        raise ValueError("select_items requires estimator='mmle'")

    active = np.ones(n_items, dtype=bool)
    rounds: list[ItemScreeningRound] = []
    removed: dict[str, list[str]] = {}
    result = None

    for round_index in range(max_rounds):
        idx = np.flatnonzero(active)
        y_r = y[:, idx]
        obs_r = observed[:, idx]
        fid_r = d_of_i[idx]
        # remap dims to a dense 0..D'-1 (dims can empty out only via floor)
        result = fit(
            np.where(obs_r, y_r, np.nan),
            fid_r,
            config,
            group_id=group_id,
            cluster_id=cluster_id,
        )
        # person screen — prior means matter for the Snijders MAP correction:
        # multilevel EAPs absorb the cluster intercepts, multigroup the group
        # means, so r_0 must be centered accordingly.
        prior_mean = None
        if result.population is not None:
            popk = result.population
            if popk["kind"] == "multilevel" and cluster_id is not None:
                u = np.asarray(popk["u_eap"], dtype=float)
                prior_mean = np.repeat(
                    u[np.asarray(cluster_id, dtype=np.int64)][:, None],
                    int(fid_r.max()) + 1,
                    axis=1,
                )
            elif popk["kind"] == "multigroup" and group_id is not None:
                prior_mean = np.asarray(popk["mu"], dtype=float)[
                    np.asarray(group_id, dtype=np.int64)
                ]
        pf = person_fit(
            np.where(obs_r, y_r, np.nan), fid_r, result.params, result.model,
            prior_mean=prior_mean, flag_threshold=person_flag_threshold,
        )
        weight = (~pf.flagged).astype(float)
        # flags
        sx2_res = s_x2(
            np.where(obs_r, y_r, np.nan),
            fid_r,
            result.params,
            result.model,
            q_theta=config.q_theta,
            q_xi=config.q_xi,
            fdr_q=fdr_q,
            person_weight=weight,
            min_effect=sx2_min_effect,
        )
        msq = infit_outfit(np.where(obs_r, y_r, np.nan), fid_r, result.params, result.model)
        a_est = np.exp(result.params.alpha)
        pos_count = np.where(obs_r, y_r, 0.0).sum(axis=0)
        neg_count = obs_r.sum(axis=0) - pos_count
        gamma = float(np.exp(result.params.tau))
        mean_dist = gamma * np.mean(
            np.sqrt(
                1e-8
                + np.sum(
                    (np.asarray(result.params.xi)[:, None, :]
                     - np.asarray(result.params.zeta)[None, :, :]) ** 2,
                    axis=2,
                )
            ),
            axis=0,
        )
        med = float(np.median(mean_dist))
        mad = float(np.median(np.abs(mean_dist - med))) * 1.4826
        iso_z = (mean_dist - med) / mad if mad > 0 else np.zeros_like(mean_dist)

        free_alpha = result.model not in {"MLSRM", "ULSRM"}
        uses_space = result.model != "MIRT"
        flags: dict[str, dict[str, bool]] = {}
        to_remove: list[int] = []
        reasons: dict[str, list[str]] = {}
        for local_i, gi in enumerate(idx):
            code = codes[gi]
            f = {
                "sparse": bool(
                    pos_count[local_i] < min_positive or neg_count[local_i] < min_positive
                ),
                "sx2": bool(sx2_res.flagged_bh[local_i]),
                "msq": bool(
                    msq["infit"][local_i] < msq_band[0]
                    or msq["infit"][local_i] > msq_band[1]
                ),
                "outfit_out": bool(
                    msq["outfit"][local_i] < msq_band[0]
                    or msq["outfit"][local_i] > msq_band[1]
                ),
                "low_disc": bool(free_alpha and a_est[local_i] < min_discrimination),
                "isolated": bool(uses_space and iso_z[local_i] > isolation_z),
            }
            flags[code] = f
            n_soft = sum(f[k] for k in ("sx2", "msq", "low_disc", "isolated"))
            if f["sparse"] or n_soft >= min_flags_to_remove:
                to_remove.append(local_i)
                reasons[code] = (["sparse"] if f["sparse"] else []) + [
                    k for k in ("sx2", "msq", "low_disc", "isolated") if f[k]
                ]

        # enforce the per-dimension floor: keep the least-flagged items
        kept_after = active.copy()
        for local_i in to_remove:
            kept_after[idx[local_i]] = False
        for d in range(int(d_of_i.max()) + 1):
            dim_items = np.flatnonzero((d_of_i == d) & active)
            surviving = np.flatnonzero((d_of_i == d) & kept_after)
            deficit = min_items_per_dim - len(surviving)
            if deficit > 0:
                dropped = [g for g in dim_items if not kept_after[g]]
                for g in dropped[:deficit]:
                    kept_after[g] = True
                    code = codes[g]
                    reasons.pop(code, None)

        removed_codes = [codes[g] for g in np.flatnonzero(active & ~kept_after)]
        rounds.append(
            ItemScreeningRound(
                round_index=round_index,
                kept_items=[codes[g] for g in np.flatnonzero(kept_after)],
                removed_items=removed_codes,
                reasons=dict(reasons),
                flags=flags,
            )
        )
        for code in removed_codes:
            removed[code] = reasons.get(code, [])
        if not removed_codes:
            break
        active = kept_after

    return ItemScreeningResult(
        kept_items=[codes[g] for g in np.flatnonzero(active)],
        removed_items=removed,
        rounds=rounds,
        final_result=result,
    )


# --------------------------------------------------------------------------
# model comparison and dimensionality residuals (Rust-core compute)
# --------------------------------------------------------------------------


def vuong_nonnested(
    loglik_a: np.ndarray,
    loglik_b: np.ndarray,
    k_a: int,
    k_b: int,
    bic_correction: bool = True,
) -> dict:
    """Vuong test for non-nested model comparison from casewise marginal
    log-likelihoods (Schneider, Chalmers, Debelak & Merkle 2019). Positive z
    favors model A; ``bic_correction`` applies the Schwarz penalty."""
    core = _core_module()
    if core is None:
        raise RuntimeError("vuong_nonnested requires the compiled Rust core")
    return dict(
        core.vuong_nonnested(
            np.asarray(loglik_a, dtype=np.float64),
            np.asarray(loglik_b, dtype=np.float64),
            int(k_a),
            int(k_b),
            bool(bic_correction),
        )
    )


def dimensionality_residuals(
    responses: np.ndarray,
    factor_id: np.ndarray,
    params,
    model: str,
    mask: np.ndarray | None = None,
    eps_distance: float = 1e-8,
) -> dict:
    """Yen Q3 residual correlations and the GDDM discrepancy (the usable
    residual-based procedures of the Svetina & Levy 2014 framework), computed
    from EAP residuals ``y - P_hat`` in the Rust core. Large |Q3| pairs signal
    unmodeled local dependence; GDDM near 0 supports the fitted structure."""
    core = _core_module()
    if core is None:
        raise RuntimeError("dimensionality_residuals requires the compiled Rust core")
    model = model.upper()
    free_alpha = model not in {"MLSRM", "ULSRM"}
    uses_space = model != "MIRT"
    y = np.asarray(responses, dtype=float)
    observed = ~np.isnan(y) if mask is None else np.asarray(mask, dtype=bool)
    d_of_i = np.asarray(factor_id, dtype=np.int64)
    a = np.exp(params.alpha) if free_alpha else np.ones(len(params.b))
    eta = a[None, :] * np.asarray(params.theta)[:, d_of_i] + params.b[None, :]
    if uses_space:
        diff = np.asarray(params.xi)[:, None, :] - np.asarray(params.zeta)[None, :, :]
        dist = np.sqrt(eps_distance + np.sum(diff * diff, axis=2))
        eta = eta - math.exp(params.tau) * dist
    p = 1.0 / (1.0 + np.exp(-np.clip(eta, -700, 700)))
    resid = np.where(observed, y - p, np.nan)
    out = dict(
        core.dimensionality_residuals(
            resid.astype(np.float64).ravel(), int(y.shape[0]), int(y.shape[1])
        )
    )
    out["q3"] = np.asarray(out["q3"])
    return out


# --------------------------------------------------------------------------
# DIF analysis: group-specific item parameters + likelihood-ratio tests
# --------------------------------------------------------------------------


@dataclass
class DIFResult:
    item_codes: list[str]
    lr_statistic: np.ndarray
    df: np.ndarray
    p_value: np.ndarray
    flagged_bh: np.ndarray
    b_by_group: np.ndarray
    a_by_group: np.ndarray
    effect_size: np.ndarray


def dif_analysis(
    responses: np.ndarray,
    factor_id: np.ndarray,
    group_id: np.ndarray,
    config=None,
    item_codes: list[str] | None = None,
    studied_items: list[int] | None = None,
    mask: np.ndarray | None = None,
    fdr_q: float = 0.05,
) -> DIFResult:
    """Likelihood-ratio DIF screen with group-specific item parameters.

    Design per Jeon, Rijmen & Rabe-Hesketh (2013; multiple-group DIF with
    group-specific item parameters and anchored impact) and Makransky & Glas
    (2013; MML DIF for the 2-PL with iterative purification): for each
    studied item, the constrained multigroup fit (common item parameters,
    group trait means/SDs free) is compared against an augmented fit in which
    that item is split into group-specific virtual items (its ``(a, b)`` free
    per group, all other items anchored at the constrained estimates).
    ``LR = 2 (ll_aug - ll_con)`` with ``df = (G - 1) x params-per-item``;
    Benjamini-Hochberg controls the FDR over studied items. The effect size
    is the largest between-group ``b`` difference on the logit scale.

    Virtual items keep the latent-space positions anchored (interaction DIF
    would be confounded with the map; see the formula compilation, part I,
    section 5.2).
    """
    from .config import FitConfig
    from .fit import fit

    y = np.asarray(responses, dtype=float)
    if mask is not None:
        y = np.where(np.asarray(mask, dtype=bool), y, np.nan)
    d_of_i = np.asarray(factor_id, dtype=np.int64)
    gid = np.asarray(group_id, dtype=np.int64)
    n_groups = int(gid.max()) + 1
    n_items = y.shape[1]
    codes = item_codes or [f"item_{i:03d}" for i in range(n_items)]
    studied = list(range(n_items)) if studied_items is None else list(studied_items)
    config = config or FitConfig(model="MLS2PLM", estimator="mmle")
    if config.estimator != "mmle":
        raise ValueError("dif_analysis requires estimator='mmle'")
    free_alpha = config.normalized_model() not in {"MLSRM", "ULSRM"}
    params_per_item = 2 if free_alpha else 1

    constrained = fit(y, d_of_i, config, group_id=gid)
    ll_con = constrained.loglik_trace[-1]

    lr = np.full(n_items, np.nan)
    dof = np.full(n_items, np.nan)
    pval = np.full(n_items, np.nan)
    b_by_group = np.full((n_items, n_groups), np.nan)
    a_by_group = np.full((n_items, n_groups), np.nan)
    effect = np.full(n_items, np.nan)

    for i in studied:
        cols = [np.where(gid == g, y[:, i], np.nan) for g in range(n_groups)]
        y_aug = np.concatenate(
            [np.delete(y, i, axis=1)] + [c[:, None] for c in cols], axis=1
        )
        fid_aug = np.concatenate(
            [np.delete(d_of_i, i), np.full(n_groups, d_of_i[i], dtype=np.int64)]
        )
        n_rest = n_items - 1
        fixed = np.zeros(n_rest + n_groups, dtype=bool)
        fixed[:n_rest] = True
        anchors = dict(
            fixed=fixed,
            alpha=np.concatenate(
                [np.delete(constrained.params.alpha, i), np.zeros(n_groups)]
            ),
            b=np.concatenate([np.delete(constrained.params.b, i), np.zeros(n_groups)]),
            zeta=np.concatenate(
                [
                    np.delete(constrained.params.zeta, i, axis=0),
                    np.repeat(constrained.params.zeta[i][None, :], n_groups, axis=0),
                ],
                axis=0,
            ),
            tau=float(constrained.params.tau),
        )
        augmented = fit(y_aug, fid_aug, config, group_id=gid, anchors=anchors)
        ll_aug = augmented.loglik_trace[-1]
        stat = max(0.0, 2.0 * (ll_aug - ll_con))
        df_i = (n_groups - 1) * params_per_item
        lr[i] = stat
        dof[i] = df_i
        pval[i] = chi2_sf(stat, df_i)
        b_g = augmented.params.b[n_rest:]
        a_g = np.exp(augmented.params.alpha[n_rest:])
        b_by_group[i] = b_g
        a_by_group[i] = a_g
        effect[i] = float(np.nanmax(b_g) - np.nanmin(b_g))

    return DIFResult(
        item_codes=codes,
        lr_statistic=lr,
        df=dof,
        p_value=pval,
        flagged_bh=benjamini_hochberg(pval, fdr_q),
        b_by_group=b_by_group,
        a_by_group=a_by_group,
        effect_size=effect,
    )


def residual_item_fit(
    responses: np.ndarray,
    factor_id: np.ndarray,
    params,
    model: str,
    mask: np.ndarray | None = None,
    n_bins: int = 10,
    eps_distance: float = 1e-8,
) -> dict:
    """Residual-based item fit (Haberman, Sinharay & Chon 2013): max |z| over
    EAP-score bins per item with Bonferroni normal p-values. Designed for
    long tests; prefer S-X2 below ~25 items (EAP shrinkage bias)."""
    core = _core_module()
    if core is None:
        raise RuntimeError("residual_item_fit requires the compiled Rust core")
    y = np.asarray(responses, dtype=float)
    observed = ~np.isnan(y) if mask is None else np.asarray(mask, dtype=bool)
    d_of_i = np.asarray(factor_id, dtype=np.int64)
    n_dims = int(d_of_i.max()) + 1
    bank = _bank_args(params, d_of_i, model, n_dims, eps_distance)
    res = dict(
        core.residual_item_fit(
            np.where(observed, y, 0.0).ravel(), observed.ravel(), int(y.shape[0]),
            bank["alpha"], bank["b"], bank["zeta"], bank["tau"], bank["factor_id"],
            bank["model"], bank["n_dims"], bank["latent_dim"], bank["eps_distance"],
            np.asarray(params.theta, dtype=np.float64).ravel(),
            np.asarray(params.xi, dtype=np.float64).ravel(),
            n_bins=int(n_bins),
        )
    )
    res["max_abs_z"] = np.asarray(res["max_abs_z"])
    res["p_value"] = np.asarray(res["p_value"])
    return res


def adjusted_chi2_pairs(
    responses: np.ndarray,
    factor_id: np.ndarray,
    params,
    model: str,
    mask: np.ndarray | None = None,
    q_theta: int = 21,
    q_xi: int = 11,
    eps_distance: float = 1e-8,
) -> dict:
    """N-adjusted pairwise chi2/df ratios (Tay & Drasgow 2012); values above
    ~3 flag pairwise misfit / local dependence."""
    core = _core_module()
    if core is None:
        raise RuntimeError("adjusted_chi2_pairs requires the compiled Rust core")
    y = np.asarray(responses, dtype=float)
    observed = ~np.isnan(y) if mask is None else np.asarray(mask, dtype=bool)
    d_of_i = np.asarray(factor_id, dtype=np.int64)
    n_dims = int(d_of_i.max()) + 1
    bank = _bank_args(params, d_of_i, model, n_dims, eps_distance)
    res = dict(
        core.adjusted_chi2_pairs(
            np.where(observed, y, 0.0).ravel(), observed.ravel(), int(y.shape[0]),
            bank["alpha"], bank["b"], bank["zeta"], bank["tau"], bank["factor_id"],
            bank["model"], bank["n_dims"], bank["latent_dim"], bank["eps_distance"],
            np.zeros(n_dims), np.ones(n_dims),
            q_theta=int(q_theta), xi_rule="gh", q_xi=int(q_xi),
        )
    )
    res["ratio"] = np.asarray(res["ratio"])
    return res


def person_fit_resampling(
    responses: np.ndarray,
    factor_id: np.ndarray,
    params,
    model: str,
    mask: np.ndarray | None = None,
    prior_mean: np.ndarray | None = None,
    n_replicates: int = 200,
    seed: int = 1,
    eps_distance: float = 1e-8,
) -> np.ndarray:
    """Parametric-bootstrap person-fit p-values (Sinharay 2016): empirical
    `P(l_z*_rep <= l_z*_obs)` per person, replicates simulated at the EAP
    estimates — robust where the asymptotic N(0,1) reference degrades."""
    core = _core_module()
    if core is None:
        raise RuntimeError("person_fit_resampling requires the compiled Rust core")
    y = np.asarray(responses, dtype=float)
    observed = ~np.isnan(y) if mask is None else np.asarray(mask, dtype=bool)
    d_of_i = np.asarray(factor_id, dtype=np.int64)
    n_dims = int(d_of_i.max()) + 1
    n_persons = y.shape[0]
    bank = _bank_args(params, d_of_i, model, n_dims, eps_distance)
    pm = None
    if prior_mean is not None:
        pm = np.broadcast_to(
            np.asarray(prior_mean, dtype=np.float64), (n_persons, n_dims)
        ).ravel().copy()
    pv = core.person_fit_resampling(
        np.where(observed, y, 0.0).ravel(), observed.ravel(), int(n_persons),
        bank["alpha"], bank["b"], bank["zeta"], bank["tau"], bank["factor_id"],
        bank["model"], bank["n_dims"], bank["latent_dim"], bank["eps_distance"],
        np.asarray(params.theta, dtype=np.float64).ravel(),
        np.asarray(params.xi, dtype=np.float64).ravel(),
        prior_mean=pm, n_replicates=int(n_replicates), seed=int(seed),
    )
    return np.asarray(pv)


def tcc_drift(
    params_old,
    params_new,
    factor_id: np.ndarray,
    model: str,
    threshold: float = 0.05,
    q_theta: int = 21,
    q_xi: int = 11,
    eps_distance: float = 1e-8,
) -> dict:
    """Stepwise TCC drift detection between two same-scale calibrations
    (Guo, Zheng & Chang 2015): flags items whose parameter drift moves the
    test characteristic curve, in removal order."""
    core = _core_module()
    if core is None:
        raise RuntimeError("tcc_drift requires the compiled Rust core")
    d_of_i = np.asarray(factor_id, dtype=np.int64)
    n_dims = int(d_of_i.max()) + 1
    old = _bank_args(params_old, d_of_i, model, n_dims, eps_distance)
    new = _bank_args(params_new, d_of_i, model, n_dims, eps_distance)
    res = dict(
        core.tcc_drift(
            old["alpha"], old["b"], old["zeta"], old["tau"],
            new["alpha"], new["b"], new["zeta"], new["tau"],
            old["factor_id"], old["model"], old["n_dims"], old["latent_dim"],
            old["eps_distance"], np.zeros(n_dims), np.ones(n_dims),
            q_theta=int(q_theta), xi_rule="gh", q_xi=int(q_xi),
            threshold=float(threshold),
        )
    )
    return res


def empirical_reliability(result) -> np.ndarray:
    """Empirical (marginal) EAP reliability per trait dimension:
    `Var(EAP) / (Var(EAP) + mean(SE^2))` (Stanley & Edwards 2016; Milanzi et
    al. 2015). Only meaningful for a well-fitting model — report alongside
    the fit statistics. Requires a marginal (MMLE) fit with posterior SDs."""
    core = _core_module()
    if core is None:
        raise RuntimeError("empirical_reliability requires the compiled Rust core")
    if result.population is None or "theta_sd" not in result.population:
        raise ValueError("empirical_reliability needs a marginal fit with theta_sd")
    theta = np.asarray(result.params.theta, dtype=np.float64)
    sd = np.asarray(result.population["theta_sd"], dtype=np.float64)
    return np.asarray(
        core.empirical_reliability(
            theta.ravel(), sd.ravel(), int(theta.shape[0]), int(theta.shape[1])
        )
    )


# --------------------------------------------------------------------------
# M2 limited-information goodness-of-fit (Maydeu-Olivares & Joe 2005, 2006;
# Cai & Hansen 2013). Rust core is the compute path; the NumPy body below is
# the parity reference and fallback.
# --------------------------------------------------------------------------


@dataclass
class M2Result:
    m2: float
    df: float
    p_value: float
    rmsea2: float
    rmsea2_ci_lower: float
    rmsea2_ci_upper: float
    srmsr: float
    n_moments: int
    n_parameters: int
    n_complete: int


class _MutBank:
    """Minimal params-like carrier for finite-difference re-evaluation."""

    __slots__ = ("alpha", "b", "zeta", "tau")

    def __init__(self, alpha, b, zeta, tau):
        self.alpha = alpha
        self.b = b
        self.zeta = zeta
        self.tau = tau


def m2(
    responses: np.ndarray,
    factor_id: np.ndarray,
    params,
    model: str,
    mask: np.ndarray | None = None,
    q_theta: int = 21,
    q_xi: int = 11,
    eps_distance: float = 1e-8,
) -> M2Result:
    """M2 statistic (order-2 residual margins), df, p-value, RMSEA2 with a 90%
    noncentral-chi-square CI, and the bivariate SRMSR. Complete cases only —
    M2 presumes a single sample size N (Maydeu-Olivares & Joe 2006)."""
    core = _core_module()
    y0 = np.asarray(responses, dtype=float)
    observed0 = ~np.isnan(y0) if mask is None else np.asarray(mask, dtype=bool)
    d_of_i = np.asarray(factor_id, dtype=np.int64)
    n_dims = int(d_of_i.max()) + 1
    if core is not None:
        bank = _bank_args(params, d_of_i, model, n_dims, eps_distance)
        res = core.m2_stat(
            np.where(observed0, y0, 0.0).ravel(),
            observed0.ravel(),
            int(y0.shape[0]),
            bank["alpha"], bank["b"], bank["zeta"], bank["tau"], bank["factor_id"],
            bank["model"], bank["n_dims"], bank["latent_dim"], bank["eps_distance"],
            np.zeros(n_dims), np.ones(n_dims),
            q_theta=int(q_theta), xi_rule="gh", q_xi=int(q_xi),
        )
        return M2Result(
            m2=float(res["m2"]), df=float(res["df"]), p_value=float(res["p_value"]),
            rmsea2=float(res["rmsea2"]),
            rmsea2_ci_lower=float(res["rmsea2_ci_lower"]),
            rmsea2_ci_upper=float(res["rmsea2_ci_upper"]),
            srmsr=float(res["srmsr"]),
            n_moments=int(res["n_moments"]), n_parameters=int(res["n_parameters"]),
            n_complete=int(res["n_complete"]),
        )
    return _m2_numpy(y0, observed0, d_of_i, params, model, q_theta, q_xi, eps_distance)


def _ncchi2_cdf(x: float, df: float, lam: float) -> float:
    if lam <= 0.0:
        return 1.0 - chi2_sf(x, df)
    half = 0.5 * lam
    term = math.exp(-half)
    total = term * (1.0 - chi2_sf(x, df))
    for j in range(1, 10000):
        term *= half / j
        total += term * (1.0 - chi2_sf(x, df + 2.0 * j))
        if term < 1e-15 and j > half:
            break
    return min(1.0, max(0.0, total))


def _nc_lambda_for(x: float, df: float, target: float) -> float:
    if (1.0 - chi2_sf(x, df)) <= target:
        return 0.0
    hi = 1.0
    while _ncchi2_cdf(x, df, hi) > target and hi < 1e8:
        hi *= 2.0
    lo = 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _ncchi2_cdf(x, df, mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _m2_numpy(y0, observed0, d_of_i, params, model, q_theta, q_xi, eps_distance):
    model_u = model.upper()
    free_alpha = model_u not in {"MLSRM", "ULSRM"}
    uses_space = model_u != "MIRT"
    n_persons, n_items = y0.shape
    latent_dim = int(np.asarray(params.zeta).shape[1])
    if n_items < 3:
        raise ValueError("M2 needs at least 3 items")

    # moment layout: univariate then bivariate pairs (i < j)
    pairs = [(i, j) for i in range(n_items) for j in range(i + 1, n_items)]
    moment_items = [[i] for i in range(n_items)] + [[i, j] for (i, j) in pairs]
    s = len(moment_items)

    # free item parameters (matching the estimator's count)
    plist = []
    for i in range(n_items):
        plist.append(("b", i, 0))
        if free_alpha:
            plist.append(("a", i, 0))
        if uses_space:
            for k in range(latent_dim):
                plist.append(("z", i, k))
    tau_free = uses_space and model_u in {"MLS2PLM", "ULS2PLM", "MLSRM", "ULSRM"}
    if tau_free:
        plist.append(("t", 0, 0))
    p = len(plist)
    if s <= p:
        raise ValueError(f"M2 df non-positive: {s} <= {p}")

    complete = np.all(observed0, axis=1)
    idx = np.where(complete)[0]
    n_c = int(idx.size)
    if n_c < p + 2:
        raise ValueError(f"too few complete cases for M2: {n_c}")
    yc = y0[idx]
    p_obs = np.empty(s)
    for i in range(n_items):
        p_obs[i] = np.mean(yc[:, i] != 0.0)
    for m, (i, j) in enumerate(pairs):
        p_obs[n_items + m] = np.mean((yc[:, i] != 0.0) & (yc[:, j] != 0.0))

    prior_mean = np.zeros(n_dims_of(d_of_i))

    def node_probs(pp):
        probs, t_w, x_w, _ = _icc_grid(pp, d_of_i, model, q_theta, q_xi, eps_distance, prior_mean)
        w = np.multiply.outer(t_w, x_w).ravel()
        return probs.reshape(n_items, -1), w

    probs0, weights = node_probs(params)

    def pi_set(probs, sset):
        pr = weights.copy()
        for m in sset:
            pr = pr * probs[m]
        return float(pr.sum())

    def model_moments(probs):
        return np.array([pi_set(probs, sset) for sset in moment_items])

    mom0 = model_moments(probs0)
    e = p_obs - mom0

    # Delta by central differences of the node moments
    alpha0 = np.asarray(params.alpha, dtype=float).copy()
    b0 = np.asarray(params.b, dtype=float).copy()
    zeta0 = np.asarray(params.zeta, dtype=float).copy()
    tau0 = float(params.tau)
    delta = np.zeros((s, p))
    for col, (kind, i, k) in enumerate(plist):
        base = {"b": b0[i], "a": alpha0[i], "z": zeta0[i, k], "t": tau0}[kind]
        h = 1e-4 * (1.0 + abs(base))
        a, b, z, t = alpha0.copy(), b0.copy(), zeta0.copy(), tau0
        if kind == "b":
            b[i] = base + h
        elif kind == "a":
            a[i] = base + h
        elif kind == "z":
            z[i, k] = base + h
        else:
            t = base + h
        mp, _ = node_probs(_MutBank(a, b, z, t))
        mom_plus = model_moments(mp)
        a, b, z, t = alpha0.copy(), b0.copy(), zeta0.copy(), tau0
        if kind == "b":
            b[i] = base - h
        elif kind == "a":
            a[i] = base - h
        elif kind == "z":
            z[i, k] = base - h
        else:
            t = base - h
        mm, _ = node_probs(_MutBank(a, b, z, t))
        mom_minus = model_moments(mm)
        delta[:, col] = (mom_plus - mom_minus) * (0.5 / h)

    # Xi_2 via the local-independence factorization of union margins
    xi = np.zeros((s, s))
    for a_i in range(s):
        for b_i in range(a_i, s):
            u = list(dict.fromkeys(moment_items[a_i] + moment_items[b_i]))
            cov = pi_set(probs0, u) - mom0[a_i] * mom0[b_i]
            xi[a_i, b_i] = cov
            xi[b_i, a_i] = cov

    n_f = float(n_c)
    u = np.linalg.solve(xi, e)          # Xi^-1 e
    w = np.linalg.solve(xi, delta)      # Xi^-1 Delta
    amat = delta.T @ w                  # Delta' Xi^-1 Delta
    g = w.T @ e                         # Delta' Xi^-1 e
    z = np.linalg.solve(amat, g)
    m2v = max(0.0, n_f * (float(e @ u) - float(g @ z)))
    df = float(s - p)
    p_value = chi2_sf(m2v, df)
    denom = df * (n_f - 1.0)
    rmsea2 = math.sqrt(max(0.0, m2v - df) / denom)
    ci_lo = math.sqrt(_nc_lambda_for(m2v, df, 0.95) / denom)
    ci_hi = math.sqrt(_nc_lambda_for(m2v, df, 0.05) / denom)

    ss, cnt = 0.0, 0
    for m, (i, j) in enumerate(pairs):
        pi, pj, pij = p_obs[i], p_obs[j], p_obs[n_items + m]
        mi, mj, mij = mom0[i], mom0[j], mom0[n_items + m]
        dobs = pi * (1 - pi) * pj * (1 - pj)
        dmod = mi * (1 - mi) * mj * (1 - mj)
        if dobs > 1e-12 and dmod > 1e-12:
            robs = (pij - pi * pj) / math.sqrt(dobs)
            rmod = (mij - mi * mj) / math.sqrt(dmod)
            ss += (robs - rmod) ** 2
            cnt += 1
    srmsr = math.sqrt(ss / cnt) if cnt else float("nan")

    return M2Result(
        m2=m2v, df=df, p_value=p_value, rmsea2=rmsea2,
        rmsea2_ci_lower=ci_lo, rmsea2_ci_upper=ci_hi, srmsr=srmsr,
        n_moments=s, n_parameters=p, n_complete=n_c,
    )


def n_dims_of(d_of_i):
    return int(np.asarray(d_of_i).max()) + 1
