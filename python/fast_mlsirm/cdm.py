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
    columns (an attribute measured by no item) are rejected. These checks do not
    establish global model identifiability: DINA identifiability requires stronger
    Q-matrix conditions (Gu & Xu, 2019), which callers must assess for their design.

    References (APA 7th ed.):
        de la Torre, J. (2009). DINA model and parameter estimation: A didactic.
            *Journal of Educational and Behavioral Statistics, 34*(1), 115–130.
            https://doi.org/10.3102/1076998607309474
        Gu, Y., & Xu, G. (2019). The sufficient and necessary condition for the
            identifiability and estimability of the DINA model. *Psychometrika,
            84*(2), 468–483. https://doi.org/10.1007/s11336-018-9619-8
        Junker, B. W., & Sijtsma, K. (2001). Cognitive assessment models with few
            assumptions, and connections with nonparametric item response theory.
            *Applied Psychological Measurement, 25*(3), 258–272.
            https://doi.org/10.1177/01466210122032064
        Templin, J. L., & Henson, R. A. (2006). Measurement of psychological
            disorders using cognitive diagnosis models. *Psychological Methods,
            11*(3), 287–305. https://doi.org/10.1037/1082-989X.11.3.287
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


@dataclass
class GdinaFit:
    """Fitted saturated G-DINA model (de la Torre, 2011).

    Item parameters are ragged: item ``i`` has ``2 ** k_required[i]`` reduced
    attribute-mastery classes, stored as the CSR slice
    ``[item_off[i]:item_off[i+1]]`` of ``item_prob`` and ``item_delta``.
    ``item_prob`` holds the free success probabilities ``P(X_i = 1 | reduced
    class l)``; ``item_delta`` the identity-link parameters (intercept, main
    effects, interactions). ``map_profile``/``attr_prob`` are the per-person MAP
    profile and marginal attribute-mastery probabilities."""

    item_off: np.ndarray
    item_prob: np.ndarray
    item_delta: np.ndarray
    k_required: np.ndarray
    profile_prob: np.ndarray
    map_profile: np.ndarray
    attr_prob: np.ndarray
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    n_parameters: int

    def item_prob_row(self, i: int) -> np.ndarray:
        """Success probabilities of item ``i``'s ``2 ** K_i`` reduced classes."""
        return self.item_prob[self.item_off[i] : self.item_off[i + 1]]

    def item_delta_row(self, i: int) -> np.ndarray:
        """Identity-link parameters of item ``i`` (intercept, mains, interactions)."""
        return self.item_delta[self.item_off[i] : self.item_off[i + 1]]


def fit_gdina(
    responses: np.ndarray,
    q_matrix: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> GdinaFit:
    """Fit the saturated G-DINA model (compute in Rust; de la Torre, 2011).

    G-DINA is the general cognitive-diagnosis framework: for item ``i`` requiring
    ``K_i`` attributes, each of the ``2 ** K_i`` reduced attribute-mastery classes
    gets a FREE success probability, estimated by marginal-ML EM over the ``2 ** K``
    profiles with the closed-form saturated M-step ``p_il = R_il / I_il`` (expected
    correct / expected total in reduced class ``l``). DINA, DINO, A-CDM, LLM and
    R-RUM are constrained special cases readable off the fitted identity-link
    ``item_delta`` (e.g. DINA leaves only the intercept and the highest-order
    interaction nonzero). ``responses`` is a persons x items 0/1 array (``NaN`` =
    missing, dropped under MAR); ``q_matrix`` is an items x attributes 0/1 array.

    References (APA 7th ed.):
        de la Torre, J. (2011). The generalized DINA model framework.
            *Psychometrika, 76*(2), 179-199.
            https://doi.org/10.1007/s11336-011-9207-7
        Ma, W., & de la Torre, J. (2020). GDINA: An R package for cognitive
            diagnosis modeling. *Journal of Statistical Software, 93*(14), 1-26.
            https://doi.org/10.18637/jss.v093.i14
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_gdina"):
        raise RuntimeError("fit_gdina requires the compiled Rust core")

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
    res = core.fit_gdina(
        yy,
        observed.reshape(-1),
        q.astype(np.int64).reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_attributes),
        int(max_iter),
        float(tol),
    )
    return GdinaFit(
        item_off=np.asarray(res["item_off"], dtype=np.int64),
        item_prob=np.asarray(res["item_prob"], dtype=np.float64),
        item_delta=np.asarray(res["item_delta"], dtype=np.float64),
        k_required=np.asarray(res["k_required"], dtype=np.int64),
        profile_prob=np.asarray(res["profile_prob"], dtype=np.float64),
        map_profile=np.asarray(res["map_profile"], dtype=np.int64),
        attr_prob=np.asarray(res["attr_prob"], dtype=np.float64).reshape(n_persons, n_attributes),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        n_parameters=int(res["n_parameters"]),
    )


@dataclass
class QMatrixValidation:
    """Result of empirical Q-matrix validation (de la Torre & Chiu, 2016).

    ``suggested_q`` is the validated items x attributes 0/1 Q-matrix — per item the
    fewest-attribute vector whose PVAF (proportion of variance accounted for)
    reaches ``epsilon``. ``suggested_pvaf``/``provisional_pvaf`` are the per-item
    PVAF of the suggested and the caller's provisional q-vector; ``flagged`` marks
    the items whose suggested vector differs from the provisional one."""

    suggested_q: np.ndarray
    suggested_pvaf: np.ndarray
    provisional_pvaf: np.ndarray
    flagged: np.ndarray
    epsilon: float


def validate_q_matrix(
    responses: np.ndarray,
    provisional_q: np.ndarray,
    epsilon: float = 0.95,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> QMatrixValidation:
    """Validate a Q-matrix by the PVAF method (compute in Rust; de la Torre & Chiu, 2016).

    The G-DINA item response function varies across the ``2^K`` latent attribute
    classes. A candidate q-vector groups those classes into masters vs. non-masters
    of its required attributes; the proportion of the item's across-class variance
    that grouping captures is its ``PVAF``. For each item the method returns the
    q-vector with the FEWEST required attributes whose ``PVAF >= epsilon``: an
    under-specified provisional vector falls short of the cutoff and is enlarged, an
    over-specified one is trimmed because a smaller vector already reaches it.

    The class distribution and identified attribute labels come from a G-DINA fit
    with the provisional Q; each item's saturated success probability over all
    ``2^K`` classes is then recovered from the fitted posteriors, so a mis-specified
    item's true attribute dependence is exposed by the attributes identified from
    the other items. The method therefore assumes the provisional Q is mostly
    correct (enough items to identify the attributes).

    ``responses`` is a persons x items 0/1 array (``NaN`` = missing, dropped under
    MAR); ``provisional_q`` is an items x attributes 0/1 array, each item loading at
    least one attribute (``K`` up to 10). ``epsilon`` is the PVAF cutoff.

    References (APA 7th ed.):
        de la Torre, J., & Chiu, C.-Y. (2016). A general method of empirical Q-matrix
            validation. *Psychometrika, 81*(2), 253-273.
            https://doi.org/10.1007/s11336-015-9467-8
        de la Torre, J. (2008). An empirically based method of Q-matrix validation
            for the DINA model: Development and applications. *Journal of Educational
            Measurement, 45*(4), 343-362.
            https://doi.org/10.1111/j.1745-3984.2008.00069.x
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "validate_q_matrix"):
        raise RuntimeError("validate_q_matrix requires the compiled Rust core")

    y = np.asarray(responses, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    q = np.asarray(provisional_q)
    if q.ndim != 2:
        raise ValueError("provisional_q must be a 2-D items x attributes array")
    n_persons, n_items = y.shape
    if q.shape[0] != n_items:
        raise ValueError("provisional_q must have one row per item")
    n_attributes = q.shape[1]

    observed = np.isfinite(y)
    yy = np.where(observed, y, 0.0).reshape(-1)
    res = core.validate_q_matrix(
        yy,
        observed.reshape(-1),
        q.astype(np.int64).reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_attributes),
        float(epsilon),
        int(max_iter),
        float(tol),
    )
    return QMatrixValidation(
        suggested_q=np.asarray(res["suggested_q"], dtype=np.int64).reshape(n_items, n_attributes),
        suggested_pvaf=np.asarray(res["suggested_pvaf"], dtype=np.float64),
        provisional_pvaf=np.asarray(res["provisional_pvaf"], dtype=np.float64),
        flagged=np.asarray(res["flagged"], dtype=bool),
        epsilon=float(res["epsilon"]),
    )


@dataclass
class WaldModelSelection:
    """Result of item-level CDM model selection by the Wald test (de la Torre, 2011).

    ``models`` names the candidate reduced models (parsimony order). ``wald_stat``,
    ``wald_df`` and ``p_value`` are items x models arrays of the Wald statistic,
    degrees of freedom, and upper-tail p-value (``NaN``/0 where a test is undefined,
    i.e. an item requiring fewer than two attributes). ``selected`` is per item the
    index into ``models`` of the chosen reduced model, or ``-1`` for the saturated
    G-DINA."""

    models: list
    wald_stat: np.ndarray
    wald_df: np.ndarray
    p_value: np.ndarray
    selected: np.ndarray
    alpha: float


def gdina_wald_selection(
    responses: np.ndarray,
    q_matrix: np.ndarray,
    alpha: float = 0.05,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> WaldModelSelection:
    """Item-level CDM model selection by the Wald test (compute in Rust; de la Torre, 2011).

    For each item the saturated G-DINA is compared with reduced models that are exact
    linear restrictions of its identity-link parameters ``delta`` (the intercept,
    main effects, and interactions of the reduced attribute-mastery classes):

    * **DINA** (conjunctive): only the intercept and the top-order interaction free.
    * **A-CDM** (additive): all interaction terms zero (intercept + main effects).

    The Wald statistic ``W = delta_R' Sigma_R^{-1} delta_R ~ chi^2(df)`` tests whether
    the restricted coordinates are jointly zero; ``Sigma_delta = M^{-1} Var(P) M^{-T}``
    is the delta-method covariance with ``Var(P_l) = P_l(1-P_l)/I_l`` (complete-data /
    expected information). Per item the fewest-parameter model with ``p > alpha`` is
    selected; if all reduced models are rejected, the saturated G-DINA is kept.

    Note: the complete-data covariance uses expected rather than observed information,
    so the test is mildly liberal (Type I slightly above ``alpha``); the gap shrinks
    with sample size and item discrimination and with strong attribute identification.
    DINO (a general linear restriction) and LLM / R-RUM (additive on other links) are
    deferred.

    ``responses`` is a persons x items 0/1 array (``NaN`` = missing, dropped under
    MAR); ``q_matrix`` is an items x attributes 0/1 array.

    References (APA 7th ed.):
        de la Torre, J. (2011). The generalized DINA model framework. *Psychometrika,
            76*(2), 179-199. https://doi.org/10.1007/s11336-011-9207-7
        Ma, W., Iaconangelo, C., & de la Torre, J. (2016). Model similarity, model
            selection, and attribute classification. *Applied Psychological
            Measurement, 40*(3), 200-217. https://doi.org/10.1177/0146621615621717
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "gdina_wald_selection"):
        raise RuntimeError("gdina_wald_selection requires the compiled Rust core")

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
    res = core.gdina_wald_selection(
        yy,
        observed.reshape(-1),
        q.astype(np.int64).reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_attributes),
        float(alpha),
        int(max_iter),
        float(tol),
    )
    models = list(res["models"])
    n_models = len(models)
    return WaldModelSelection(
        models=models,
        wald_stat=np.asarray(res["wald_stat"], dtype=np.float64).reshape(n_items, n_models),
        wald_df=np.asarray(res["wald_df"], dtype=np.int64).reshape(n_items, n_models),
        p_value=np.asarray(res["p_value"], dtype=np.float64).reshape(n_items, n_models),
        selected=np.asarray(res["selected"], dtype=np.int64),
        alpha=float(res["alpha"]),
    )
