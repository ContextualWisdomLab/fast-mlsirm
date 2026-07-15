"""Mixed Rasch / mixture IRT (Rost, 1990): the population is a mixture of latent
classes, each with its own item parameters, fit by marginal-ML EM in the Rust core."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MixtureFit:
    """Fitted mixture-IRT model (Rost, 1990).

    ``a``/``b`` are the per-class item discriminations and difficulties, shape
    ``(n_classes, n_items)`` (``a`` is all ones for the Rasch model); ``pi`` the
    mixing proportions; ``class_posterior`` the ``(n_persons, n_classes)`` class
    responsibilities ``P(class | x_j)``; ``map_class`` the per-person modal class;
    ``theta`` the mixture-EAP ability. Classes are in canonical order (mixing weight
    descending, ties broken by mean difficulty ascending)."""

    model: str
    n_classes: int
    a: np.ndarray
    b: np.ndarray
    pi: np.ndarray
    class_posterior: np.ndarray
    map_class: np.ndarray
    theta: np.ndarray
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    n_parameters: int


def fit_mixture(
    responses: np.ndarray,
    n_classes: int = 2,
    model: str = "rasch",
    n_starts: int = 1,
    max_iter: int = 500,
    tol: float = 1e-6,
    seed: int = 0x2545F491,
) -> MixtureFit:
    """Fit a mixed Rasch / mixture-IRT model (compute in Rust; Rost, 1990).

    The population is modeled as a mixture of ``n_classes`` latent classes, each with
    its own item parameters and a mixing proportion, detecting unobserved
    heterogeneity (qualitatively different response strategies). Within a class,
    responses follow a Rasch (``model="rasch"``, discrimination fixed at 1) or 2PL
    (``model="2pl"``) model with ability ``theta ~ N(0, 1)``, estimated by marginal-ML
    EM. Because the mixture likelihood is multimodal, pass ``n_starts > 1`` to run
    several restarts and keep the highest-likelihood fit (start 0 is a deterministic
    warm start). ``responses`` is a persons x items 0/1 array (``NaN`` = missing,
    dropped under MAR). Classes are returned in a canonical order.

    Note: this is the marginal-ML / ``N(0,1)`` operationalization (Rost & von Davier,
    1995; the form in psychomix, Frick et al., 2012), which yields item contrasts
    equivalent to Rost's (1990) original conditional-ML formulation under a different
    location convention.

    References (APA 7th ed.):
        Rost, J. (1990). Rasch models in latent classes: An integration of two
            approaches to item analysis. *Applied Psychological Measurement, 14*(3),
            271-282. https://doi.org/10.1177/014662169001400305
        Rost, J., & von Davier, M. (1995). Mixture distribution Rasch models. In G. H.
            Fischer & I. W. Molenaar (Eds.), *Rasch models* (pp. 257-268). Springer.
        Frick, H., Strobl, C., Leisch, F., & Zeileis, A. (2012). Flexible Rasch
            mixture models with package psychomix. *Journal of Statistical Software,
            48*(7), 1-25. https://doi.org/10.18637/jss.v048.i07
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_mixture"):
        raise RuntimeError("fit_mixture requires the compiled Rust core")

    y = np.asarray(responses, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    observed = np.isfinite(y)
    yy = np.where(observed, y, 0.0).reshape(-1)
    res = core.fit_mixture(
        yy,
        observed.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_classes),
        str(model),
        int(n_starts),
        int(max_iter),
        float(tol),
        int(seed),
    )
    c = int(res["n_classes"])
    return MixtureFit(
        model=str(res["model"]),
        n_classes=c,
        a=np.asarray(res["a"], dtype=np.float64).reshape(c, n_items),
        b=np.asarray(res["b"], dtype=np.float64).reshape(c, n_items),
        pi=np.asarray(res["pi"], dtype=np.float64),
        class_posterior=np.asarray(res["class_posterior"], dtype=np.float64).reshape(n_persons, c),
        map_class=np.asarray(res["map_class"], dtype=np.int64),
        theta=np.asarray(res["theta"], dtype=np.float64),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        n_parameters=int(res["n_parameters"]),
    )
