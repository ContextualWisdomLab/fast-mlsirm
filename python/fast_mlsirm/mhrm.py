"""Metropolis-Hastings Robbins-Monro (MH-RM) confirmatory multidimensional 2PL (Cai, 2010).

A stochastic-approximation EM that scales confirmatory item factor analysis to a latent
dimensionality where the deterministic Gauss-Hermite / quasi-Monte-Carlo E-steps of
:func:`fast_mlsirm.fit_2pl` become infeasible. Each cycle imputes the traits with a short persistent
random-walk Metropolis chain, then takes one Robbins-Monro stochastic-Newton step on the
block-diagonal (per-item) complete-data score and information. Orthogonal factors (``Sigma = I``).
The numerical estimation runs in Rust."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import ConfirmatoryModel, ExploratoryModel, IrtModel, _resolve_model

_MAX_DIMS = 64


@dataclass
class MhrmFit:
    """Fitted MH-RM confirmatory multidimensional 2PL (Cai, 2010).

    ``loading`` is the ``n_items x n_dims`` matrix of free loadings ``a_id`` (exactly ``0`` where the
    confirmatory pattern is ``0``), per-dimension reflection-canonicalized so each dimension's largest
    pure anchor loads positive; ``intercept`` the per-item ``b_i``; ``theta`` the ``n_persons x
    n_dims`` trait EAP (Monte-Carlo mean of the imputed draws over the convergence stage); ``corr`` the
    ``n_dims x n_dims`` latent correlation matrix ``Phi`` (identity when ``estimate_corr=False``, unit
    diagonal with estimated off-diagonals otherwise); ``se_loading`` / ``se_intercept`` the Louis
    (1982) observed-information standard errors (empty when ``estimate_se=False``; a block falls back to
    the complete-data Fisher information where the finite-sample Louis block is not positive-definite).
    The model is ``P(X_ij=1 | theta_j) = sigmoid(sum_d a_id theta_jd + b_i)`` with
    ``theta_j ~ MVN(0, Phi)``.
    ``acceptance_rate`` is the final tuned Metropolis acceptance; ``termination_reason`` is
    ``"converged"`` or ``"max_cycles_reached"``; ``final_param_change`` the windowed mean parameter
    change at termination."""

    model: IrtModel
    loading: np.ndarray
    intercept: np.ndarray
    theta: np.ndarray
    corr: np.ndarray
    se_loading: np.ndarray
    se_intercept: np.ndarray
    acceptance_rate: float
    n_cycles: int
    converged: bool
    termination_reason: str
    final_param_change: float
    n_parameters: int

    @property
    def n_dims(self) -> int:
        """Latent dimension count derived from :attr:`model`."""

        return self.model.n_dims


def fit_mhrm(
    responses: np.ndarray,
    model: int | ExploratoryModel | ConfirmatoryModel = 1,
    max_cycles: int = 2000,
    burn_in: int = 200,
    mh_steps: int = 5,
    proposal_sd: float = 1.0,
    target_accept: float = 0.30,
    tol: float = 1e-3,
    seed: int = 0x9E37_79B9_7F4A_7C15,
    estimate_se: bool = True,
    estimate_corr: bool = False,
) -> MhrmFit:
    """Fit the confirmatory multidimensional 2PL by Metropolis-Hastings Robbins-Monro (compute in
    Rust; Cai, 2010).

    A stochastic-approximation EM for the general compensatory 2PL,
    ``P(X_ij=1 | theta_j) = sigmoid(sum_{d in S_i} a_id theta_jd + b_i)`` with ``theta_j ~ MVN(0,
    I_D)``. Unlike :func:`fast_mlsirm.fit_2pl`, the marginal-likelihood integral is not quadratured:
    each cycle (1) imputes each person's ``theta`` by a short PERSISTENT (warm-started) random-walk
    Metropolis chain from its current posterior, and (2) takes one Robbins-Monro stochastic-Newton
    step ``xi <- xi + gain_k Gamma_k^{-1} s_k`` on the block-diagonal per-item complete-data score
    ``s_k`` and Robbins-Monro-smoothed information ``Gamma_k``. The gain follows a constant-gain
    burn-in then a decreasing ``1/(k - burn_in)`` schedule (``sum gain = inf``, ``sum gain^2 < inf``),
    converging almost surely to a marginal-score root. Because the per-item work is closed-form and
    ``D``-independent, MH-RM scales to a latent dimensionality (``n_dims`` up to 64) where the
    ``q**n_dims`` Gauss-Hermite grid and the QMC E-step are infeasible.

    Identification: unit trait variances fix the loading scale; a PURE single-dimension anchor item
    per dimension pins the rotation; the per-dimension sign is CANONICALIZED (largest pure anchor
    positive), enforced in-loop each cycle so the stochastic running average stays in one mirror mode.
    Loadings are UNCONSTRAINED so reverse-keyed / negative cross-loadings are representable. Standard
    errors are the Louis (1982) observed information accumulated over the convergence stage.

    ``responses`` is a persons x items 0/1 array (``NaN`` = missing, dropped under MAR). For
    ``model=1`` all item loadings on the single factor are free; a multidimensional confirmatory
    structure is supplied with ``model=models.confirmatory(loading_pattern)``; every dimension needs a
    pure single-loading anchor item. ``burn_in`` must be less than ``max_cycles``; ``proposal_sd`` is
    the initial random-walk SD, auto-tuned toward ``target_accept`` during burn-in. With
    ``estimate_corr=True`` a free latent CORRELATION matrix ``Phi`` (``theta ~ MVN(0, Phi)``, unit
    diagonal) is estimated by a per-cycle Robbins-Monro gradient step (Cai, 2010b); with ``False``
    (default) the factors are orthogonal (``Phi = I``) and the fit is bit-identical to the flag off.

    References (APA 7th ed.):
        Cai, L. (2010). High-dimensional exploratory item factor analysis by a Metropolis-Hastings
            Robbins-Monro algorithm. *Psychometrika, 75*(1), 33-57.
            https://doi.org/10.1007/s11336-009-9136-x
        Cai, L. (2010). Metropolis-Hastings Robbins-Monro algorithm for confirmatory item factor
            analysis. *Journal of Educational and Behavioral Statistics, 35*(3), 307-335.
            https://doi.org/10.3102/1076998609353115
        Louis, T. A. (1982). Finding the observed information matrix when using the EM algorithm.
            *Journal of the Royal Statistical Society: Series B, 44*(2), 226-233.
            https://doi.org/10.1111/j.2517-6161.1982.tb01203.x
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_mhrm"):
        raise RuntimeError("fit_mhrm requires the compiled Rust core")

    y = np.asarray(responses, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    resolved_model, pat = _resolve_model(model, n_items)
    n_dims = pat.shape[1]
    if not 1 <= n_dims <= _MAX_DIMS:
        raise ValueError(f"loading_pattern dimensions must be between 1 and {_MAX_DIMS}")

    def _finite_int(value, name: str) -> int:
        scalar = np.asarray(value)
        if (
            scalar.ndim != 0
            or not np.issubdtype(scalar.dtype, np.number)
            or np.iscomplexobj(scalar)
        ):
            raise ValueError(f"{name} must be a finite integer")
        numeric = float(scalar)
        if not np.isfinite(numeric) or numeric != np.floor(numeric):
            raise ValueError(f"{name} must be a finite integer")
        return int(numeric)

    max_cycles_int = _finite_int(max_cycles, "max_cycles")
    burn_in_int = _finite_int(burn_in, "burn_in")
    mh_steps_int = _finite_int(mh_steps, "mh_steps")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed_int = int(seed)
    if not 0 <= seed_int < 2**64:
        raise ValueError("seed must be in [0, 2**64)")
    for name, val in (("proposal_sd", proposal_sd), ("target_accept", target_accept), ("tol", tol)):
        if not np.isfinite(float(val)):
            raise ValueError(f"{name} must be finite")

    observed = ~np.isnan(y)
    if np.any(observed):
        obs_y = y[observed]
        if np.any((obs_y != 0) & (obs_y != 1)):
            raise ValueError("responses must be 0, 1, or NaN (missing)")
    yy = np.where(observed, y, 0.0).astype(np.int64).reshape(-1)

    res = core.fit_mhrm(
        yy,
        observed.reshape(-1),
        pat.astype(np.int64).reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_dims),
        max_cycles_int,
        burn_in_int,
        mh_steps_int,
        float(proposal_sd),
        float(target_accept),
        float(tol),
        seed_int,
        bool(estimate_se),
        bool(estimate_corr),
    )
    se_loading = np.asarray(res["se_loading"], dtype=np.float64)
    se_intercept = np.asarray(res["se_intercept"], dtype=np.float64)
    return MhrmFit(
        model=resolved_model,
        loading=np.asarray(res["loading"], dtype=np.float64).reshape(n_items, n_dims),
        intercept=np.asarray(res["intercept"], dtype=np.float64),
        theta=np.asarray(res["theta"], dtype=np.float64).reshape(n_persons, n_dims),
        corr=np.asarray(res["corr"], dtype=np.float64).reshape(n_dims, n_dims),
        se_loading=se_loading.reshape(n_items, n_dims) if se_loading.size else se_loading,
        se_intercept=se_intercept,
        acceptance_rate=float(res["acceptance_rate"]),
        n_cycles=int(res["n_cycles"]),
        converged=bool(res["converged"]),
        termination_reason=str(res["termination_reason"]),
        final_param_change=float(res["final_param_change"]),
        n_parameters=int(res["n_parameters"]),
    )
