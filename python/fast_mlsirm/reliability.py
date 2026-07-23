"""Guttman lambda reliability coefficients (Guttman, 1945, as implemented by
CRAN psych 2.6.5; Revelle, 2025). All numeric work happens in the Rust core;
this module only validates and marshals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GuttmanResult:
    """Guttman lambda coefficients plus split-half summaries.

    ``lambda4`` is the best (maximum) split-half over the enumerated or
    sampled splits, ``beta`` the worst (minimum, floored at 0), and
    ``mean_split`` the average; ``n_splits`` reports how many splits were
    evaluated and ``exhaustive`` whether all C(p, floor(p/2)) subsets were
    enumerated."""

    lambda1: float
    lambda2: float
    lambda3: float
    lambda4: float
    lambda5: float
    lambda6: float
    beta: float
    mean_split: float
    n_splits: int
    exhaustive: bool


_REFERENCES = """References (APA 7th ed.):
        Guttman, L. (1945). A basis for analyzing test-retest reliability.
            *Psychometrika, 10*(4), 255-282.
            https://doi.org/10.1007/BF02288892 (as cited in Revelle, 2025)
        Revelle, W. (2025). *psych: Procedures for psychological,
            psychometric, and personality research* (Version 2.6.5)
            [R package]. https://CRAN.R-project.org/package=psych
    """


def guttman_lambdas(
    data: np.ndarray,
    n_sample_splits: int = 15000,
    seed: int = 1,
) -> GuttmanResult:
    """Guttman's lambda reliability coefficients (compute in Rust; algorithm
    transcribed from the psych 2.6.5 R sources ``guttman.R``, ``splitHalf.R``
    and ``smc.R``, read line by line; Guttman, 1945, not read — attribution
    as cited in Revelle, 2025).

    Computed on the Pearson correlation matrix of ``data`` (an
    ``n_persons x n_items`` array, complete and finite): lambda1-lambda3
    (lambda3 is coefficient alpha), lambda5 (best single covariance column),
    lambda6 (squared multiple correlations), and split-half summaries
    lambda4 (best split), beta (worst split), and the mean split. All
    ``floor(p/2)``-subsets are enumerated when their count fits within
    ``n_sample_splits`` (psych's brute-force cutoff is 15000); otherwise
    that many splits are sampled with this crate's deterministic LCG, so
    sampled results are psych-inspired but not bit-identical to any R run.
    Declared divergences from psych (documented in the Rust module): no
    ``check.keys`` auto-reversal, absolute split-half correlations in both
    branches, plain matrix inverse with an error on singular correlation
    matrices instead of a pseudoinverse. In LLM-as-a-Judge quality
    management these bound the internal consistency of a judge rubric.

    """ + _REFERENCES
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "guttman_lambdas"):
        raise RuntimeError("guttman_lambdas requires the compiled Rust core")
    x = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
    if x.ndim != 2:
        raise ValueError("data must be a 2-D persons x items array")
    n_persons, n_items = x.shape
    if int(n_sample_splits) < 1:
        raise ValueError("n_sample_splits must be >= 1")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    res = core.guttman_lambdas(
        x.reshape(-1), int(n_persons), int(n_items), int(n_sample_splits), int(seed)
    )
    return GuttmanResult(
        lambda1=float(res["lambda1"]),
        lambda2=float(res["lambda2"]),
        lambda3=float(res["lambda3"]),
        lambda4=float(res["lambda4"]),
        lambda5=float(res["lambda5"]),
        lambda6=float(res["lambda6"]),
        beta=float(res["beta"]),
        mean_split=float(res["mean_split"]),
        n_splits=int(res["n_splits"]),
        exhaustive=bool(res["exhaustive"]),
    )
