"""Rust-owned, metric-explicit reporting transforms for orthogonal GRMs."""

import numpy as np


def orthogonal_grm_report(slopes, intercepts, *, input_metric, scale):
    """Transform one item's parameters for independent unit-variance factors.

    ``slopes`` and strictly decreasing boundary ``intercepts`` are nonempty,
    real, one-dimensional NumPy arrays. ``input_metric`` is ``normal_ogive``
    (requiring ``scale=1``) or ``logistic_approximation`` (requiring an explicit
    positive finite conversion scale). Logistic conversion is approximate;
    no universal scale or covariance-general interpretation is supplied.
    All loadings, thresholds, communality and uniqueness are computed in Rust.

    Basis: Muraki and Carlson (1995, pp. 74–76, eqs. 23–25; p. 79, eqs. 38–39)
    and Chalmers (2012, p. 3, eq. 1; p. 20, footnote 3).

    References
    ----------
    Muraki, E., & Carlson, J. E. (1995). Full-information factor analysis for
    polytomous item responses. Applied Psychological Measurement, 19(1), 73–90.
    https://doi.org/10.1177/014662169501900109
    Chalmers, R. P. (2012). mirt: A multidimensional item response theory package
    for the R environment. Journal of Statistical Software, 48(6), 1–29.
    https://doi.org/10.18637/jss.v048.i06
    """
    if type(input_metric) is not str or input_metric not in (
        "normal_ogive", "logistic_approximation"
    ):
        raise ValueError("input_metric must be normal_ogive or logistic_approximation")
    if type(scale) not in (int, float, np.float32, np.float64, np.int32, np.int64):
        raise ValueError("scale must be a concrete real scalar")
    try:
        scale = float(scale)
    except (ValueError, OverflowError):
        raise ValueError("scale must be finite and positive") from None
    arrays = []
    for name, value in (("slopes", slopes), ("intercepts", intercepts)):
        if type(value) is not np.ndarray or value.ndim != 1 or value.dtype.kind not in "fiu":
            raise ValueError(f"{name} must be a real one-dimensional NumPy array")
        arrays.append(np.ascontiguousarray(value, dtype=np.float64))
    from . import _core

    result = _core.orthogonal_grm_report(*arrays, input_metric, scale)
    result["loadings"] = np.asarray(result["loadings"], dtype=np.float64)
    result["thresholds"] = np.asarray(result["thresholds"], dtype=np.float64)
    return result
