"""Haberman subscore added-value analysis via proportional reduction in mean
squared error (PRMSE; Haberman, 2008, as cited in Sinharay, 2010). All numeric
work happens in the Rust core; this module only validates and marshals
arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SubscoreResult:
    """Haberman subscore added-value analysis for ``K`` subscales.

    ``alpha`` are the per-subscale Cronbach alphas ( = ``prmse_s``);
    ``alpha_total`` the total-test alpha. ``corr`` is the ``(K+1) x (K+1)``
    correlation matrix of the observed subscores with the total score last;
    ``disattenuated_corr`` the ``K x K`` disattenuated subscore correlations
    (NaN diagonal). ``prmse_s``/``prmse_x``/``prmse_sx`` are the PRMSEs of
    predicting the true subscore from the observed subscore, the observed
    total, and both; ``tau``/``beta``/``gamma`` the augmented-regression
    weights. ``added_value_s`` is Haberman's rule ``PRMSE_s > PRMSE_x``;
    ``added_value_sx`` uses Sinharay's (2010) ``+ 0.01`` margin.
    ``observed`` (``n x K``), ``total`` (``n``), and the three estimator
    matrices ``subscore_s``/``subscore_x``/``subscore_sx`` (each ``n x K``)
    give per-person scores."""

    alpha: np.ndarray
    alpha_total: float
    corr: np.ndarray
    disattenuated_corr: np.ndarray
    prmse_s: np.ndarray
    prmse_x: np.ndarray
    prmse_sx: np.ndarray
    tau: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    added_value_s: np.ndarray
    added_value_sx: np.ndarray
    observed: np.ndarray
    total: np.ndarray
    subscore_s: np.ndarray
    subscore_x: np.ndarray
    subscore_sx: np.ndarray


def subscore_analysis(
    responses: np.ndarray,
    groups: np.ndarray,
) -> SubscoreResult:
    """Haberman subscore added-value analysis (compute in Rust; Haberman,
    2008, as cited in Sinharay, 2010).

    Decides, for each subscale of a test, whether reporting its subscore adds
    value over reporting the total score alone, by comparing the PRMSEs of
    three classical-test-theory estimators of the true subscore (from the
    observed subscore, from the observed total, and from both jointly).
    Formulas follow the Appendix of Sinharay (2010) and the CRAN ``subscore``
    package R source (both read); Haberman (2008) and Wainer et al. (2001)
    are cited only through Sinharay (2010). Degenerate samples (any Cronbach
    alpha outside ``(0, 1]``, zero-variance scores, a subscore collinear with
    the total) are rejected with ``ValueError`` rather than propagating NaN.

    In LLM-as-a-Judge item-quality management this decides whether
    per-domain judge subscores carry diagnostic information beyond the
    overall score, or whether reporting them would be statistically
    misleading.

    ``responses`` is a complete ``persons x items`` array of scored
    responses (``n >= 3``). ``groups`` assigns each item an integer subscale
    index in ``0..K`` (``K >= 2``, every subscale with at least 2 items,
    partition exhaustive by construction).

    References (APA 7th ed.):
        Haberman, S. J. (2008). When can subscores have value? *Journal of
            Educational and Behavioral Statistics, 33*(2), 204-229.
            https://doi.org/10.3102/1076998607302636 (as cited in Sinharay,
            2010)
        Sinharay, S. (2010). *When can subscores be expected to have added
            value? Results from operational and simulated data* (ETS
            Research Rep. No. RR-10-16). Educational Testing Service.
        Wainer, H., Vevea, J. L., Camacho, F., Reeve, B. B., Rosa, K., &
            Nelson, L. (2001). Augmented scores — "borrowing strength" to
            compute scores based on small numbers of items. In D. Thissen &
            H. Wainer (Eds.), *Test scoring* (pp. 343-387). Lawrence
            Erlbaum. (as cited in Sinharay, 2010)
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "subscore_analysis"):
        raise RuntimeError("subscore_analysis requires the compiled Rust core")

    y = np.asarray(responses, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    if n_persons < 3 or n_items < 4:
        # K >= 2 subscales with >= 2 items each needs at least 4 items;
        # rejecting here keeps degenerate shapes (e.g. huge zero-column
        # arrays) from crossing the Rust boundary at all.
        raise ValueError("responses needs at least 3 persons and 4 items")
    if not np.all(np.isfinite(y)):
        raise ValueError("responses must be complete (no missing values)")

    g = np.asarray(groups).reshape(-1)
    if g.shape[0] != n_items:
        raise ValueError("groups must assign one subscale index per item")
    if not np.issubdtype(g.dtype, np.integer):
        gf = np.asarray(groups, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(gf)) or np.any(gf != np.round(gf)):
            raise ValueError("groups must be integer subscale indices")
        g = gf.astype(np.int64)
    if np.any(g < 0):
        raise ValueError("groups must be nonnegative subscale indices")
    if np.any(g >= n_items):
        # trust boundary: the subscale count drives Rust-side allocations
        raise ValueError("groups indices must be < n_items")

    res = core.subscore_analysis(
        y.reshape(-1), int(n_persons), int(n_items), [int(v) for v in g]
    )
    k = len(res["alpha"])
    return SubscoreResult(
        alpha=np.asarray(res["alpha"], dtype=np.float64),
        alpha_total=float(res["alpha_total"]),
        corr=np.asarray(res["corr"], dtype=np.float64).reshape(k + 1, k + 1),
        disattenuated_corr=np.asarray(
            res["disattenuated_corr"], dtype=np.float64
        ).reshape(k, k),
        prmse_s=np.asarray(res["prmse_s"], dtype=np.float64),
        prmse_x=np.asarray(res["prmse_x"], dtype=np.float64),
        prmse_sx=np.asarray(res["prmse_sx"], dtype=np.float64),
        tau=np.asarray(res["tau"], dtype=np.float64),
        beta=np.asarray(res["beta"], dtype=np.float64),
        gamma=np.asarray(res["gamma"], dtype=np.float64),
        added_value_s=np.asarray(res["added_value_s"], dtype=bool),
        added_value_sx=np.asarray(res["added_value_sx"], dtype=bool),
        observed=np.asarray(res["observed"], dtype=np.float64).reshape(
            n_persons, k
        ),
        total=np.asarray(res["total"], dtype=np.float64),
        subscore_s=np.asarray(res["subscore_s"], dtype=np.float64).reshape(
            n_persons, k
        ),
        subscore_x=np.asarray(res["subscore_x"], dtype=np.float64).reshape(
            n_persons, k
        ),
        subscore_sx=np.asarray(res["subscore_sx"], dtype=np.float64).reshape(
            n_persons, k
        ),
    )
