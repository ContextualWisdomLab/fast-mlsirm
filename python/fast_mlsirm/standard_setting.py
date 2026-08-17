"""Standard-setting methods. All numeric work happens in the Rust core;
this module only validates and marshals arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HofsteeResult:
    """Hofstee compromise standard-setting result.

    ``cut_score`` and ``fail_rate`` are the coordinates of the Hofstee
    point (percent units). ``failed`` is True when the cumulative score
    ogive and the Hofstee diagonal do not cross and the ported R fallback
    produced the point (cut pinned to ``min_cut`` or ``max_cut`` with a
    two-decimal directed-rounded fail rate). ``cum_freq_percent`` is the
    cumulative relative frequency in percent at integer scores 0..=100."""

    cut_score: float
    fail_rate: float
    failed: bool
    cum_freq_percent: np.ndarray


def hofstee(
    scores,
    min_cut: float,
    max_cut: float,
    min_fail: float,
    max_fail: float,
) -> HofsteeResult:
    """Hofstee compromise cut score
    (``mlsirm_core::standard_setting::hofstee``), a computational port of
    the psychometricsGP R package's ``fn_plot_hofstee()`` (READ:
    ``R/fn_plot_hofstee.R``, the only inspectable implementation found;
    single-source port, stated openly; plotting excluded). NOT READ:
    Hofstee (1983), the originating paper — the method is cited only as
    implemented by psychometricsGP.

    The cut score is the intersection of the piecewise-linear cumulative
    relative frequency curve over integer score bins 0..=100 (a score in
    ``(s-1, s]`` counts in bin ``s``) with the descending diagonal from
    ``(min_cut, max_fail)`` to ``(max_cut, min_fail)``. When they do not
    cross, the R fallback pins the cut to ``min_cut`` (if the strict
    below-``min_cut`` rate exceeds ``max_fail``; rate rounded UP to two
    decimals) or ``max_cut`` (rate rounded DOWN), with ``failed=True``.

    Reduced scope (adversarial spec review): collinear overlap between
    the ogive and the diagonal, and a zero-length diagonal
    (``min_cut == max_cut and min_fail == max_fail``), raise
    ``ValueError`` — spatstat ``crossing.psp`` degenerate semantics were
    not verified against an R runtime.

    In LLM-as-a-Judge quality management this sets a defensible pass/fail
    threshold on judge or item scores from panel-style bounds on the
    acceptable cut score and failure rate.

    ``scores`` is a 1-D vector of percentages in [0, 100]; the four
    bounds must be finite, in [0, 100], with ``min_cut <= max_cut`` and
    ``min_fail <= max_fail``.

    References
    ----------
    Hofstee, W. K. B. (1983). The case for compromise in educational
    selection and grading. In S. B. Anderson & J. S. Helmick (Eds.),
    *On educational testing* (pp. 109-127). Jossey-Bass. (NOT READ.)
    Roberts, M. (2024). *psychometricsGP* (R package),
    ``fn_plot_hofstee()``. https://github.com/GergoIO/psychometricsGP
    (READ: ``R/fn_plot_hofstee.R``; ported implementation.)
    """
    s = np.asarray(scores)
    if s.ndim != 1:
        raise ValueError("scores must be a 1-D vector")
    if s.shape[0] < 1:
        raise ValueError("scores must be non-empty")
    if np.iscomplexobj(s):
        raise ValueError("scores must be real-valued")
    if s.dtype.kind not in ("i", "u", "f"):
        raise ValueError("scores must be an integer or float array")
    sf = np.ascontiguousarray(s, dtype=np.float64)
    for name, p in (
        ("min_cut", min_cut),
        ("max_cut", max_cut),
        ("min_fail", min_fail),
        ("max_fail", max_fail),
    ):
        if not isinstance(p, (int, float)) or isinstance(p, bool):
            raise ValueError(f"{name} must be a real number")

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "py_hofstee"):
        raise RuntimeError("Rust core with py_hofstee is required")
    res = core.py_hofstee(
        sf, float(min_cut), float(max_cut), float(min_fail), float(max_fail)
    )
    return HofsteeResult(
        cut_score=float(res["cut_score"]),
        fail_rate=float(res["fail_rate"]),
        failed=bool(res["failed"]),
        cum_freq_percent=np.asarray(res["cum_freq_percent"], dtype=np.float64),
    )
