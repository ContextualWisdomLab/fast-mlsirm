"""Cognitive diagnosis models: DINA (conjunctive / AND gate) and DINO
(disjunctive / OR gate), estimated by marginal-ML EM over the ``2^K`` binary
attribute-mastery profiles in the Rust core."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CdmFit:
    """Fitted DINA/DINO cognitive diagnosis model.

    ``slip``/``guess`` are the per-item ``s_i = P(X=0 | mastered)`` and
    ``g_i = P(X=1 | not mastered)``; ``profile_prob`` the population probability of
    each of the ``2^K`` attribute profiles (bit-encoded: attribute ``k`` is mastered
    in profile ``c`` iff ``(c >> k) & 1``); ``map_profile`` the per-person posterior
    mode (bit-encoded); ``attr_prob`` the persons x attributes marginal mastery
    probabilities ``P(alpha_jk = 1 | X_j)`` (attribute EAP)."""

    model: str
    slip: np.ndarray
    guess: np.ndarray
    profile_prob: np.ndarray
    map_profile: np.ndarray
    attr_prob: np.ndarray
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    n_parameters: int

    def attribute_mastery(self) -> np.ndarray:
        """Hard 0/1 attribute-mastery classification (``attr_prob >= 0.5``)."""
        return (self.attr_prob >= 0.5).astype(np.int64)

    def profile_bits(self) -> np.ndarray:
        """Decode ``map_profile`` into a persons x attributes 0/1 matrix."""
        k = self.attr_prob.shape[1]
        codes = self.map_profile.astype(np.int64)
        return ((codes[:, None] >> np.arange(k)) & 1).astype(np.int64)


def fit_cdm(
    responses: np.ndarray,
    q_matrix: np.ndarray,
    model: str = "dina",
    max_iter: int = 500,
    tol: float = 1e-6,
) -> CdmFit:
    """Fit a DINA or DINO cognitive diagnosis model (compute in Rust).

    Each respondent has a binary attribute-mastery profile ``alpha in {0,1}^K``; the
    Q-matrix specifies which of the ``K`` attributes each item requires. The ideal
    (latent) response is ``eta = prod_k alpha_k^{q_k}`` for DINA (mastery of ALL
    required attributes) or ``eta = 1 - prod_k (1 - alpha_k)^{q_k}`` for DINO (ANY
    required attribute); the observed response adds a per-item slip and guess,
    ``P(X=1 | alpha) = (1 - s)^{eta} g^{1 - eta}``. Parameters are estimated by
    marginal-ML EM over the ``2^K`` profiles with a free profile distribution;
    persons are classified by their posterior-mode profile (``map_profile``) and
    marginal attribute probabilities (``attr_prob``).

    ``responses`` is a persons x items array of 0/1 (``NaN`` marks a missing cell,
    dropped under a missing-at-random assumption). ``q_matrix`` is an items x
    attributes 0/1 array; all-zero rows (an item measuring nothing) and all-zero
    columns (an attribute measured by no item, hence non-identified) are rejected.

    References (APA 7th ed.):
        de la Torre, J. (2009). DINA model and parameter estimation: A didactic.
            *Journal of Educational and Behavioral Statistics, 34*(1), 115-130.
            https://doi.org/10.3102/1076998607309474
        Junker, B. W., & Sijtsma, K. (2001). Cognitive assessment models with few
            assumptions, and connections with nonparametric item response theory.
            *Applied Psychological Measurement, 25*(3), 258-272.
            https://doi.org/10.1177/01466210122032064
        Templin, J. L., & Henson, R. A. (2006). Measurement of psychological
            disorders using cognitive diagnosis models. *Psychological Methods,
            11*(3), 287-305. https://doi.org/10.1037/1082-989X.11.3.287
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_cdm"):
        raise RuntimeError("fit_cdm requires the compiled Rust core")

    y = np.asarray(responses, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    q = np.asarray(q_matrix)
    if q.ndim != 2:
        raise ValueError("q_matrix must be a 2-D items x attributes array")
    n_persons, n_items = y.shape
    if q.shape[0] != n_items:
        raise ValueError("q_matrix must have one row per item")
    n_attributes = q.shape[1]

    observed = np.isfinite(y)
    yy = np.where(observed, y, 0.0).reshape(-1)
    res = core.fit_cdm(
        yy,
        observed.reshape(-1),
        q.astype(np.int64).reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_attributes),
        str(model),
        int(max_iter),
        float(tol),
    )
    return CdmFit(
        model=str(res["model"]),
        slip=np.asarray(res["slip"], dtype=np.float64),
        guess=np.asarray(res["guess"], dtype=np.float64),
        profile_prob=np.asarray(res["profile_prob"], dtype=np.float64),
        map_profile=np.asarray(res["map_profile"], dtype=np.int64),
        attr_prob=np.asarray(res["attr_prob"], dtype=np.float64).reshape(n_persons, n_attributes),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        n_parameters=int(res["n_parameters"]),
    )
