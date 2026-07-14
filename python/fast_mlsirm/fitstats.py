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
) -> SX2Result:
    """Orlando-Thissen S-X² per item, summed scores within each trait dim.

    Persons with any missing response inside a dimension are excluded from
    that dimension's observed table (the summed score would not be
    comparable). ``person_weight`` (0/1) can down-weight aberrant respondents
    flagged by person fit before item decisions (design doc §6).
    """
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
            for gn, gr, ge in groups:
                if gn <= 0:
                    continue
                e_prop = ge / gn
                if e_prop <= 0.0 or e_prop >= 1.0:
                    continue
                o_prop = gr / gn
                x2 += gn * (o_prop - e_prop) ** 2 / (e_prop * (1.0 - e_prop))
                n_grp += 1
            df_i = n_grp - n_free
            stat[i] = x2
            n_groups_out[i] = n_grp
            if df_i >= 1:
                dof[i] = df_i
                pval[i] = chi2_sf(x2, df_i)
    return SX2Result(
        statistic=stat,
        df=dof,
        p_value=pval,
        flagged_bh=benjamini_hochberg(pval, fdr_q),
        n_score_groups=n_groups_out,
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
    msq_band: tuple[float, float] = (0.7, 1.3),
    min_discrimination: float = 0.35,
    isolation_z: float = 3.0,
    min_items_per_dim: int = 4,
    max_rounds: int = 5,
    min_flags_to_remove: int = 2,
) -> ItemScreeningResult:
    """Iterative fit -> flag -> remove -> refit item screening.

    Flags per round (literature-grounded; see the formula compilation §9):

    1. ``sparse``: fewer than ``min_positive`` positive (or negative)
       observed responses — removed on this flag alone (the item cannot
       support its parameters).
    2. ``sx2``: S-X² significant after Benjamini-Hochberg at ``fdr_q``.
    3. ``msq``: infit or outfit outside ``msq_band`` (Wright & Linacre 1994).
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
        # person screen
        pf = person_fit(np.where(obs_r, y_r, np.nan), fid_r, result.params, result.model)
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
                    or msq["outfit"][local_i] < msq_band[0]
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
