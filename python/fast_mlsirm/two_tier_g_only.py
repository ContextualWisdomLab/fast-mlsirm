"""Explicit G-only expected totals; native numerical ownership.

The existing two-primary plug-in API remains a different estimand. This API
integrates W and specifics at a caller-declared nuisance prior. Substituting
G EAP gives F(E[G|responses]), not E[F(G)|responses]. No posterior averaging,
fit, convergence certification, or CPU numerical fallback occurs here.

Basis: cumulative GRM probabilities (Cai, 2010, p. 589, Eqs. 11-12) and
independent-specific integration (Cai, 2015, pp. 542-543, Eqs. 14-17).
The mean uses linearity of expectation; shared-factor score covariance and
full score distributions are not computed.

Cai, L. (2010). A two-tier full-information item factor analysis model with
applications. Psychometrika, 75(4), 581-612.
https://doi.org/10.1007/s11336-010-9178-0
Cai, L. (2015). Lord-Wingersky algorithm version 2.0 for hierarchical item
factor models with applications in test scoring, scale alignment, and model
fit testing. Psychometrika, 80(2), 535-559.
https://doi.org/10.1007/s11336-014-9411-3
"""
from __future__ import annotations
import numpy as np
from .two_tier_grm import _specific_map_control


def expected_raw_g_only(fit, specific_map, *, mean_w, sd_w, specific_sd,
                        q_wording, q_specific, device):
    """Return native scores and dispatch provenance for F(g)=E[T|G=g].

    ``fit.theta_p_eap[:, 0]`` supplies G; the W EAP column is never used.
    ``fit.phi`` must be exactly identity and primary order must be (G, W).
    Nuisance distribution settings are explicit, independent of fit quadrature.
    The category total is zero-based. Caller owns fit convergence/adoption.
    """
    if device != "gpu":
        raise ValueError("G-only scoring requires explicit GPU; fallback forbidden")
    for name, value in (("q_wording", q_wording), ("q_specific", q_specific)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive int")
    if fit.n_primary != 2 or not np.array_equal(np.asarray(fit.phi), np.eye(2)):
        raise ValueError("G-only scoring requires two orthogonal primary factors")
    a = np.asarray(fit.a_primary, dtype=np.float64)
    s = np.asarray(fit.a_specific, dtype=np.float64)
    t = np.asarray(fit.threshold, dtype=np.float64)
    theta = np.asarray(fit.theta_p_eap, dtype=np.float64)
    sd = np.asarray(specific_sd, dtype=np.float64)
    if s.ndim != 1 or a.shape != (s.size, 2) or t.shape != (s.size, fit.n_cat - 1):
        raise ValueError("item bank shapes are inconsistent")
    if theta.ndim != 2 or theta.shape[1] != 2 or not theta.shape[0]:
        raise ValueError("theta_p_eap must have shape (n_persons, 2)")
    if sd.shape != (fit.n_specific,):
        raise ValueError("specific_sd must have shape (n_specific,)")
    smap = _specific_map_control(specific_map, s.size, fit.n_specific)
    from . import _core
    if not hasattr(_core, "two_tier_expected_raw_g_only"):
        raise RuntimeError("G-only scoring requires a native core with this API")
    out = _core.two_tier_expected_raw_g_only(
        np.ascontiguousarray(a).reshape(-1), np.ascontiguousarray(s),
        np.ascontiguousarray(t).reshape(-1), np.ascontiguousarray(theta[:, 0]),
        np.ascontiguousarray(smap), fit.n_cat, fit.n_specific, mean_w, sd_w,
        np.ascontiguousarray(sd), 0., q_wording, q_specific, device)
    if (out["scores"].shape != (theta.shape[0],) or
            not np.isfinite(out["scores"]).all() or out["dispatches"] != 1 or
            out["cpu_numerical_fallbacks"] != 0 or out["estimand"] != "E[T|G=g]" or
            out["q_wording"] != q_wording or out["q_specific"] != q_specific or
            not out["backend"] or not out["device"]):
        raise RuntimeError("invalid native G-only score receipt")
    return out
