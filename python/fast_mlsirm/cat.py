"""Computerized adaptive testing (CAT) administration on a calibrated bank.

This module is a *downstream application* built on already-calibrated item
parameters. It does not touch the calibration objective, likelihood, gradients,
or the MLS2PLM formula (those are reserved for a model-design PR per
``AGENTS.md``); it only consumes the item parameters and reuses the repository's
existing probability and item-information machinery
(:func:`fast_mlsirm.diagnostics.predict_proba` and
:func:`fast_mlsirm.test_design.item_information`).

Scope and dimensionality choice
-------------------------------
Under the repository's simple-structure parameterization each item loads on a
single trait dimension ``factor_id[i]`` and the latent-space distance term
depends only on the (fixed) person/item positions ``xi``/``zeta``, not on the
trait ``theta``. Consequently ``d eta_i / d theta_d = a_i`` for the loaded
dimension and ``0`` otherwise, so the Fisher information and the ability score
equation are **block-separable across trait dimensions**. This module therefore
estimates the trait vector one dimension at a time from the items loading on it,
holding the latent-space position at the bank's population-mean ``xi`` (the same
default :func:`fast_mlsirm.test_design.item_information` uses). This is the
documented "scalar / dominant-dimension information consistent with the
parameterization" choice; a full multidimensional (directional-information)
administration is a possible follow-up.

The dichotomous item information reused here is the 2PL Fisher information
``I_j(theta) = a_j^2 P_j(theta) Q_j(theta)`` (van der Linden & Pashley, 2010).

References (APA 7th ed.)
------------------------
Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability in a
    microcomputer environment. *Applied Psychological Measurement, 6*(4),
    431-444. https://doi.org/10.1177/014662168200600405
Lord, F. M. (1980). *Applications of item response theory to practical testing
    problems*. Lawrence Erlbaum.
van der Linden, W. J., & Pashley, P. J. (2010). Item selection and ability
    estimation in adaptive testing. In W. J. van der Linden & C. A. W. Glas
    (Eds.), *Elements of adaptive testing* (pp. 3-30). Springer.
    https://doi.org/10.1007/978-0-387-85461-8_1
Weiss, D. J., & Kingsbury, G. G. (1984). Application of computerized adaptive
    testing to educational problems. *Journal of Educational Measurement,
    21*(4), 361-375. https://doi.org/10.1111/j.1745-3984.1984.tb01040.x
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .diagnostics import predict_proba
from .objective import validate_factor_id
from .test_design import item_information, select_cat_item
from .types import MLSIRMParams

# Re-export the canonical dichotomous item-information and MFI selection helpers
# so the adaptive-testing surface is discoverable from a single module. These are
# the exact functions used throughout the repository; CAT does not reimplement
# the numeric core.
__all__ = [
    "AbilityEstimate",
    "AdaptiveTestResult",
    "ability_standard_error",
    "administer_adaptive_test",
    "estimate_ability_eap",
    "estimate_ability_mle",
    "item_information",
    "select_cat_item",
    "select_max_information_item",
    "simulate_adaptive_test",
]

_PROB_EPS = 1e-12


@dataclass
class AbilityEstimate:
    """A trait estimate after a set of administered responses.

    Attributes
    ----------
    theta:
        Estimated trait vector, one entry per trait dimension.
    se:
        Per-dimension standard error. For EAP this is the posterior standard
        deviation; for MLE it is ``1 / sqrt(sum_j I_j(theta_hat))`` over the
        administered items loading on that dimension.
    method:
        ``"eap"`` or ``"mle"``.
    finite:
        Per-dimension flag. Always ``True`` for EAP (a proper prior keeps the
        posterior finite); ``False`` for an MLE dimension whose administered
        responses are all identical, for which the maximum-likelihood estimate
        has no finite root and the value is clamped to the search bound
        (Warm, 1989).
    """

    theta: np.ndarray
    se: np.ndarray
    method: str
    finite: np.ndarray


@dataclass
class AdaptiveTestResult:
    """The outcome of a single adaptive administration.

    Attributes
    ----------
    administered:
        Item indices in administration order.
    responses:
        The 0/1 responses in the same order as ``administered``.
    theta:
        Final trait estimate.
    se:
        Final per-dimension standard error.
    theta_trace:
        Trait estimate after each administered response (a list of vectors).
    se_trace:
        Standard-error vector after each administered response.
    n_items:
        Number of administered items.
    stop_reason:
        ``"max_items"`` or ``"se_threshold"``.
    method:
        Ability-estimation method used (``"eap"`` or ``"mle"``).
    """

    administered: np.ndarray
    responses: np.ndarray
    theta: np.ndarray
    se: np.ndarray
    theta_trace: list[np.ndarray] = field(default_factory=list)
    se_trace: list[np.ndarray] = field(default_factory=list)
    n_items: int = 0
    stop_reason: str = ""
    method: str = "eap"


def _bank_dims(bank: MLSIRMParams) -> int:
    """Return the number of trait dimensions carried by a calibrated bank."""
    return int(np.asarray(bank.theta).shape[1])


def _mean_xi(bank: MLSIRMParams) -> np.ndarray:
    """Return the bank's population-mean latent-space position as a row vector."""
    return np.asarray(bank.xi, dtype=np.float64).mean(axis=0, keepdims=True)


def _query_params(bank: MLSIRMParams, theta_rows: np.ndarray) -> MLSIRMParams:
    """Build query parameters that evaluate ``theta_rows`` on the bank items.

    ``theta_rows`` has shape ``(n_query, n_dims)``. The latent-space position is
    held at the bank's population-mean ``xi`` (broadcast to every query row), so
    the probabilities and item information match
    :func:`fast_mlsirm.test_design.item_information`.
    """
    theta_rows = np.asarray(theta_rows, dtype=np.float64)
    if theta_rows.ndim != 2 or theta_rows.shape[1] != _bank_dims(bank):
        raise ValueError("theta_rows must have shape (n_query, n_dims)")
    xi_rows = np.repeat(_mean_xi(bank), theta_rows.shape[0], axis=0)
    return MLSIRMParams(
        theta=theta_rows,
        alpha=bank.alpha,
        b=bank.b,
        xi=xi_rows,
        zeta=bank.zeta,
        tau=bank.tau,
    )


def _validate_administration(
    bank: MLSIRMParams,
    factor_id: np.ndarray,
    administered: np.ndarray,
    responses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize a partial administration.

    Returns the validated ``factor_id``, integer ``administered`` indices, and
    float 0/1 ``responses``.
    """
    n_items = int(np.asarray(bank.b).shape[0])
    fid = validate_factor_id(factor_id, n_items, _bank_dims(bank))
    adm = np.asarray(administered, dtype=np.int64)
    resp = np.asarray(responses, dtype=np.float64)
    if adm.ndim != 1 or resp.ndim != 1 or adm.shape != resp.shape:
        raise ValueError("administered and responses must be 1D arrays of equal length")
    if adm.size and (np.any(adm < 0) or np.any(adm >= n_items)):
        raise ValueError("administered item index out of range")
    if adm.size != len(set(adm.tolist())):
        raise ValueError("administered items must be unique")
    if resp.size and not np.all((resp == 0.0) | (resp == 1.0)):
        raise ValueError("responses must be 0 or 1")
    return fid, adm, resp


def estimate_ability_mle(
    bank: MLSIRMParams,
    factor_id: np.ndarray,
    administered: np.ndarray,
    responses: np.ndarray,
    *,
    model: str = "MLS2PLM",
    start: np.ndarray | None = None,
    max_iter: int = 50,
    tol: float = 1e-6,
    bound: float = 6.0,
) -> AbilityEstimate:
    """Maximum-likelihood trait estimate via per-dimension Newton steps.

    Solves the 2PL score equation ``sum_j a_j (u_j - P_j(theta)) = 0`` for each
    trait dimension using Newton-Raphson with the Fisher information as the
    curvature (Lord, 1980; van der Linden & Pashley, 2010). The estimate is
    clamped to ``[-bound, bound]``; a dimension whose administered responses are
    all identical has no finite MLE and is flagged ``finite=False``.
    """
    fid, adm, resp = _validate_administration(bank, factor_id, administered, responses)
    n_dims = _bank_dims(bank)
    a = np.asarray(bank.a, dtype=np.float64)
    theta = (
        np.zeros(n_dims)
        if start is None
        else np.array(start, dtype=np.float64).reshape(n_dims)
    )
    dims_present = np.unique(fid[adm]) if adm.size else np.array([], dtype=np.int64)

    for _ in range(max_iter):
        prob = np.clip(
            predict_proba(_query_params(bank, theta[None, :]), fid, model=model)[0],
            _PROB_EPS,
            1 - _PROB_EPS,
        )
        delta_max = 0.0
        for d in dims_present:
            on_d = fid[adm] == d
            sel = adm[on_d]
            a_d = a[sel]
            p_d = prob[sel]
            score = float(np.vdot(a_d, resp[on_d] - p_d))
            info = float(np.vdot(a_d * a_d, p_d * (1.0 - p_d)))
            if info <= _PROB_EPS:
                continue
            new = float(np.clip(theta[d] + score / info, -bound, bound))
            delta_max = max(delta_max, abs(new - theta[d]))
            theta[d] = new
        if delta_max < tol:
            break

    se = np.full(n_dims, np.inf)
    finite = np.ones(n_dims, dtype=bool)
    prob = np.clip(
        predict_proba(_query_params(bank, theta[None, :]), fid, model=model)[0],
        _PROB_EPS,
        1 - _PROB_EPS,
    )
    for d in dims_present:
        on_d = fid[adm] == d
        sel = adm[on_d]
        a_d = a[sel]
        p_d = prob[sel]
        info = float(np.vdot(a_d * a_d, p_d * (1.0 - p_d)))
        if info > _PROB_EPS:
            se[d] = 1.0 / np.sqrt(info)
        u_d = resp[on_d]
        if u_d.size and np.all(u_d == u_d[0]):
            finite[d] = False
            se[d] = np.inf
    return AbilityEstimate(theta=theta, se=se, method="mle", finite=finite)


def estimate_ability_eap(
    bank: MLSIRMParams,
    factor_id: np.ndarray,
    administered: np.ndarray,
    responses: np.ndarray,
    *,
    model: str = "MLS2PLM",
    prior_mean: float | np.ndarray = 0.0,
    prior_sd: float | np.ndarray = 1.0,
    n_quad: int = 41,
    quad_range: float = 6.0,
) -> AbilityEstimate:
    """Expected-a-posteriori trait estimate on a fixed grid (Bock & Mislevy, 1982).

    For each trait dimension the posterior over a ``n_quad``-node grid on
    ``[-quad_range, quad_range] + prior_mean`` is formed from a normal prior and
    the 2PL likelihood of the administered items loading on that dimension; the
    posterior mean and standard deviation are returned. The EAP estimate is
    always finite, including all-correct / all-incorrect patterns. A dimension
    with no administered items returns the prior mean and prior SD.
    """
    fid, adm, resp = _validate_administration(bank, factor_id, administered, responses)
    if not isinstance(n_quad, (int, np.integer)) or n_quad < 2:
        raise ValueError("n_quad must be an integer >= 2")
    if not (quad_range > 0):
        raise ValueError("quad_range must be positive")
    n_dims = _bank_dims(bank)
    pm = np.broadcast_to(np.asarray(prior_mean, dtype=np.float64), (n_dims,)).astype(
        np.float64
    )
    psd = np.broadcast_to(np.asarray(prior_sd, dtype=np.float64), (n_dims,)).astype(
        np.float64
    )
    if np.any(psd <= 0):
        raise ValueError("prior_sd must be positive")

    theta = pm.copy()
    se = psd.copy()
    dims_present = np.unique(fid[adm]) if adm.size else np.array([], dtype=np.int64)
    for d in dims_present:
        nodes = np.linspace(-quad_range, quad_range, int(n_quad)) + pm[d]
        theta_grid = np.repeat(theta[None, :], nodes.size, axis=0)
        theta_grid[:, d] = nodes
        prob = np.clip(
            predict_proba(_query_params(bank, theta_grid), fid, model=model),
            _PROB_EPS,
            1 - _PROB_EPS,
        )
        on_d = fid[adm] == d
        sel = adm[on_d]
        u_d = resp[on_d]
        loglik = np.sum(
            u_d[None, :] * np.log(prob[:, sel])
            + (1.0 - u_d)[None, :] * np.log(1.0 - prob[:, sel]),
            axis=1,
        )
        logprior = -0.5 * ((nodes - pm[d]) / psd[d]) ** 2
        log_post = loglik + logprior
        weights = np.exp(log_post - np.max(log_post))
        total = float(np.sum(weights))
        if total <= 0 or not np.isfinite(total):
            continue
        weights /= total
        mean = float(np.vdot(nodes, weights))
        var = float(np.vdot((nodes - mean) ** 2, weights))
        theta[d] = mean
        se[d] = float(np.sqrt(max(var, 0.0)))
    return AbilityEstimate(
        theta=theta, se=se, method="eap", finite=np.ones(n_dims, dtype=bool)
    )


def ability_standard_error(
    bank: MLSIRMParams,
    factor_id: np.ndarray,
    theta: np.ndarray,
    *,
    administered: np.ndarray | None = None,
    model: str = "MLS2PLM",
) -> np.ndarray:
    """Per-dimension asymptotic SE ``1/sqrt(sum_j I_j(theta))``.

    Information is summed over ``administered`` items loading on each dimension
    (or all items when ``administered`` is ``None``), using the reused 2PL
    Fisher information (van der Linden & Pashley, 2010). Dimensions with no
    contributing items return ``inf``.
    """
    n_dims = _bank_dims(bank)
    theta_vec = np.asarray(theta, dtype=np.float64).reshape(n_dims)
    info = item_information(
        _query_params(bank, theta_vec[None, :]), factor_id, model=model
    )
    fid = validate_factor_id(factor_id, info.size, n_dims)
    if administered is None:
        mask = np.ones(info.size, dtype=bool)
    else:
        mask = np.zeros(info.size, dtype=bool)
        adm = np.asarray(administered, dtype=np.int64)
        if adm.size and (np.any(adm < 0) or np.any(adm >= info.size)):
            raise ValueError("administered item index out of range")
        mask[adm] = True
    se = np.full(n_dims, np.inf)
    for d in range(n_dims):
        total = float(np.sum(info[mask & (fid == d)]))
        if total > _PROB_EPS:
            se[d] = 1.0 / np.sqrt(total)
    return se


def select_max_information_item(
    bank: MLSIRMParams,
    factor_id: np.ndarray,
    theta: np.ndarray,
    *,
    administered: np.ndarray | None = None,
    model: str = "MLS2PLM",
) -> int:
    """Return the maximum-Fisher-information unadministered item at ``theta``.

    Thin, keyword-friendly wrapper over
    :func:`fast_mlsirm.test_design.select_cat_item` implementing the classic
    maximum-information item-selection rule (Lord, 1980; van der Linden &
    Pashley, 2010).
    """
    theta_vec = np.asarray(theta, dtype=np.float64).reshape(_bank_dims(bank))
    return select_cat_item(
        bank, factor_id, theta=theta_vec, administered=administered, model=model
    )


def administer_adaptive_test(
    bank: MLSIRMParams,
    factor_id: np.ndarray,
    respond: Callable[[int], int],
    *,
    model: str = "MLS2PLM",
    ability_method: str = "eap",
    prior_mean: float | np.ndarray = 0.0,
    prior_sd: float | np.ndarray = 1.0,
    max_items: int | None = None,
    se_threshold: float | None = None,
    min_items: int = 1,
    n_quad: int = 41,
    quad_range: float = 6.0,
) -> AdaptiveTestResult:
    """Run a maximum-information adaptive administration to a stopping rule.

    Each step selects the maximum-Fisher-information unadministered item at the
    current trait estimate, obtains a 0/1 response from ``respond(item_index)``,
    and re-estimates the trait (EAP by default, or MLE). Administration stops
    when ``max_items`` have been given or, once at least ``min_items`` have been
    given, when the worst-dimension standard error falls to ``se_threshold`` --
    the fixed-length and fixed-precision rules of Weiss and Kingsbury (1984).

    ``respond`` is a caller-supplied callable returning the examinee's 0/1
    response for a given item index (e.g. a live examinee, or the model-based
    simulator used by :func:`simulate_adaptive_test`).
    """
    if ability_method not in {"eap", "mle"}:
        raise ValueError("ability_method must be 'eap' or 'mle'")
    n_items = int(np.asarray(bank.b).shape[0])
    fid = validate_factor_id(factor_id, n_items, _bank_dims(bank))
    if max_items is None:
        max_items = n_items
    max_items = int(max_items)
    if not (1 <= max_items <= n_items):
        raise ValueError("max_items must be between 1 and the number of items")
    if min_items < 1:
        raise ValueError("min_items must be >= 1")
    if se_threshold is not None and not (se_threshold > 0):
        raise ValueError("se_threshold must be positive")
    n_dims = _bank_dims(bank)

    def _estimate(adm: np.ndarray, resp: np.ndarray) -> AbilityEstimate:
        """Estimate the trait from the current partial administration."""
        if ability_method == "eap":
            return estimate_ability_eap(
                bank,
                fid,
                adm,
                resp,
                model=model,
                prior_mean=prior_mean,
                prior_sd=prior_sd,
                n_quad=n_quad,
                quad_range=quad_range,
            )
        return estimate_ability_mle(bank, fid, adm, resp, model=model)

    administered: list[int] = []
    responses: list[float] = []
    theta_trace: list[np.ndarray] = []
    se_trace: list[np.ndarray] = []
    theta = np.broadcast_to(np.asarray(prior_mean, dtype=np.float64), (n_dims,)).astype(
        np.float64
    )
    se = np.full(n_dims, np.inf)
    stop_reason = "max_items"

    while True:
        if len(administered) >= max_items:
            stop_reason = "max_items"
            break
        if (
            se_threshold is not None
            and len(administered) >= min_items
            and np.all(np.isfinite(se))
            and float(np.max(se)) <= se_threshold
        ):
            stop_reason = "se_threshold"
            break
        item = select_cat_item(
            bank,
            fid,
            theta=theta,
            administered=np.asarray(administered, dtype=np.int64)
            if administered
            else None,
            model=model,
        )
        u = respond(int(item))
        u = float(bool(u)) if isinstance(u, bool) else float(u)
        if u not in (0.0, 1.0):
            raise ValueError("respond must return 0 or 1")
        administered.append(int(item))
        responses.append(u)
        est = _estimate(
            np.asarray(administered, dtype=np.int64),
            np.asarray(responses, dtype=np.float64),
        )
        theta = est.theta
        se = est.se
        theta_trace.append(theta.copy())
        se_trace.append(se.copy())

    return AdaptiveTestResult(
        administered=np.asarray(administered, dtype=np.int64),
        responses=np.asarray(responses, dtype=np.float64),
        theta=theta,
        se=se,
        theta_trace=theta_trace,
        se_trace=se_trace,
        n_items=len(administered),
        stop_reason=stop_reason,
        method=ability_method,
    )


def simulate_adaptive_test(
    bank: MLSIRMParams,
    factor_id: np.ndarray,
    true_theta: np.ndarray,
    *,
    model: str = "MLS2PLM",
    seed: int = 0,
    ability_method: str = "eap",
    prior_mean: float | np.ndarray = 0.0,
    prior_sd: float | np.ndarray = 1.0,
    max_items: int | None = None,
    se_threshold: float | None = None,
    min_items: int = 1,
    n_quad: int = 41,
    quad_range: float = 6.0,
) -> AdaptiveTestResult:
    """Simulate an adaptive administration for a known ``true_theta``.

    Responses are drawn Bernoulli from the calibrated bank's own item response
    probabilities at ``true_theta`` (holding the latent-space position at the
    bank mean), then :func:`administer_adaptive_test` runs the maximum-
    information CAT loop. Deterministic given ``seed``. Useful for recovery
    studies and as a runnable demonstration.
    """
    n_dims = _bank_dims(bank)
    true_vec = np.asarray(true_theta, dtype=np.float64).reshape(n_dims)
    fid = validate_factor_id(factor_id, int(np.asarray(bank.b).shape[0]), n_dims)
    prob_true = np.clip(
        predict_proba(_query_params(bank, true_vec[None, :]), fid, model=model)[0],
        0.0,
        1.0,
    )
    rng = np.random.default_rng(int(seed))

    def _respond(item_index: int) -> int:
        """Draw a Bernoulli 0/1 response for ``item_index`` at ``true_theta``."""
        return int(rng.random() < prob_true[item_index])

    return administer_adaptive_test(
        bank,
        fid,
        _respond,
        model=model,
        ability_method=ability_method,
        prior_mean=prior_mean,
        prior_sd=prior_sd,
        max_items=max_items,
        se_threshold=se_threshold,
        min_items=min_items,
        n_quad=n_quad,
        quad_range=quad_range,
    )
