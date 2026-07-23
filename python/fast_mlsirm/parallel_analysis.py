"""Horn's parallel analysis for principal-component retention (Horn, 1965,
as implemented by CRAN paran; Dinno, 2018). All numeric work happens in the
Rust core; this module only validates and marshals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ParallelAnalysisResult:
    """Parallel-analysis outputs, all vectors in descending observed-
    eigenvalue order. ``retained`` counts leading components whose adjusted
    eigenvalue stays above 1 up to the first failure (later resurgences do
    not count, matching paran's scan)."""

    retained: int
    eigenvalues: np.ndarray
    random_eigenvalues: np.ndarray
    bias: np.ndarray
    adjusted_eigenvalues: np.ndarray


_REFERENCES = """References (APA 7th ed.):
        Dinno, A. (2018). *paran: Horn's test of principal components/
            factors* (Version 1.5.6) [R package].
            https://CRAN.R-project.org/package=paran
        Glorfeld, L. W. (1995). An improvement on Horn's parallel analysis
            methodology for selecting the correct number of factors to
            retain. *Educational and Psychological Measurement, 55*(3),
            377-393. (as cited in Dinno, 2018)
        Horn, J. L. (1965). A rationale and a test for the number of factors
            in factor analysis. *Psychometrika, 30*(2), 179-185.
            https://doi.org/10.1007/BF02289447 (as cited in Dinno, 2018)
    """


def parallel_analysis(
    data: np.ndarray,
    n_iterations: int | None = None,
    centile: int = 0,
    seed: int = 1,
) -> ParallelAnalysisResult:
    """Horn's parallel analysis, PCA path (compute in Rust; algorithm
    transcribed from the paran 1.5.6 R source, read line by line; Horn,
    1965, and Glorfeld, 1995, not read — attribution as cited in Dinno,
    2018).

    Eigenvalues of the Pearson correlation matrix of ``data`` (an
    ``n_persons x n_items`` array, complete and finite) are adjusted by the
    sampling bias estimated from ``n_iterations`` random standard-normal
    data sets of the same shape: ``adjusted = observed - (random - 1)``.
    Components are retained while ``adjusted > 1``, scanning left to right
    and stopping at the first failure. ``centile=0`` benchmarks against the
    per-position mean of the random eigenvalues (Horn's method as
    implemented by paran); ``centile`` in 1..=99 uses that upper centile
    (R type-7 quantile) instead — Glorfeld's conservative variant.
    ``n_iterations`` defaults to ``30 * n_items`` (paran's default). The
    random stream is this crate's deterministic LCG — results are
    paran-inspired but not bit-identical to any R run. In LLM-as-a-Judge
    quality management this estimates how many latent dimensions the judge
    rubric actually measures.

    """ + _REFERENCES
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "parallel_analysis"):
        raise RuntimeError("parallel_analysis requires the compiled Rust core")
    x = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
    if x.ndim != 2:
        raise ValueError("data must be a 2-D persons x items array")
    n_persons, n_items = x.shape
    iters = 30 * n_items if n_iterations is None else int(n_iterations)
    if iters < 1:
        raise ValueError("n_iterations must be >= 1")
    if not 0 <= int(centile) <= 99:
        raise ValueError("centile must be 0 (mean) or in 1..=99")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    res = core.parallel_analysis(
        x.reshape(-1), int(n_persons), int(n_items), iters, int(centile), int(seed)
    )
    return ParallelAnalysisResult(
        retained=int(res["retained"]),
        eigenvalues=np.asarray(res["eigenvalues"], dtype=np.float64),
        random_eigenvalues=np.asarray(res["random_eigenvalues"], dtype=np.float64),
        bias=np.asarray(res["bias"], dtype=np.float64),
        adjusted_eigenvalues=np.asarray(
            res["adjusted_eigenvalues"], dtype=np.float64
        ),
    )
