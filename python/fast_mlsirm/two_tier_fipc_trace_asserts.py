"""Consumer-side asserts for two-tier FIPC remapped-vs-fixed measure consistency.

Catches the stall/recovery failure mode observed at tip ``0621169b`` where
``fixed_loglik_trace`` stayed flat (evidence value ≈ ``-1405.8755``) while
``prior_mean_trace`` kept moving and ``n_accepted_prior_steps`` climbed
(evidence ``n_accepted=100``) — consistent with mean-only accepts that do not
improve the frozen-measure diagnostic objective.

These helpers are pure NumPy; they do not call the Rust core. Writers own
``two_tier_grm.rs``; consumers use this gate on exported traces.
"""

from __future__ import annotations

import numpy as np

# Evidence anchor from FIPC tip 0621169b independent read / writer traces.
EVIDENCE_FLAT_FIXED_LL_0621169B = -1405.8755


class TwoTierFipcFixedMeasureRegressionError(AssertionError):
    """Raised when prior accepts disagree with a flat fixed-measure LL."""


def _as_1d_finite(name: str, values: object) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size < 1:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _mean_path(
    prior_mean_trace: object, *, n_primary: int, n_iter: int
) -> np.ndarray:
    flat = _as_1d_finite("prior_mean_trace", prior_mean_trace)
    expected = int(n_iter) * int(n_primary)
    if flat.size != expected:
        raise ValueError(
            f"prior_mean_trace length {flat.size} != n_iter*n_primary={expected}"
        )
    return flat.reshape(int(n_iter), int(n_primary))


def fixed_loglik_post_init_is_flat(
    fixed_loglik_trace: object, *, atol: float = 1e-9
) -> bool:
    """True when every post-init fixed LL matches the first post-init value."""
    fixed = _as_1d_finite("fixed_loglik_trace", fixed_loglik_trace)
    # Index 0 is the first recorded evaluation; "post-init" = from the first
    # value onward when only one exists, else from index 1 (after the initial
    # baseline) so a single jump at iter0→1 does not mask later flatness.
    post = fixed if fixed.size == 1 else fixed[1:]
    if post.size == 0:
        return True
    return bool(np.max(np.abs(post - post[0])) <= float(atol))


def prior_mean_trace_moves(
    prior_mean_trace: object,
    *,
    n_primary: int,
    n_iter: int,
    move_atol: float = 1e-12,
) -> bool:
    """True when any iteration's mean vector differs from the first."""
    path = _mean_path(prior_mean_trace, n_primary=n_primary, n_iter=n_iter)
    if path.shape[0] < 2:
        return False
    deltas = np.linalg.norm(path - path[0], axis=1)
    return bool(np.max(deltas) > float(move_atol))


def fixed_loglik_improves(
    fixed_loglik_trace: object, *, atol: float = 1e-9
) -> bool:
    """True if some later fixed LL exceeds the first post-init value by > atol."""
    fixed = _as_1d_finite("fixed_loglik_trace", fixed_loglik_trace)
    post = fixed if fixed.size == 1 else fixed[1:]
    if post.size <= 1:
        return False
    return bool(np.max(post) > post[0] + float(atol))


def assert_two_tier_fipc_prior_accepts_improve_fixed_measure(
    *,
    fixed_loglik_trace: object,
    prior_mean_trace: object,
    n_primary: int,
    n_iter: int,
    n_accepted_prior_steps: int,
    flat_atol: float = 1e-9,
    move_atol: float = 1e-12,
    mean_only_accepts: int | None = None,
) -> None:
    """Fail closed on mean-move / accept-count without frozen-LL improvement.

    Raises
    ------
    TwoTierFipcFixedMeasureRegressionError
        (1) ``prior_mean_trace`` moves across iterations while
        ``fixed_loglik_trace`` is flat after the first post-init value; or
        (2) accepted prior steps increase on a mean-only path (explicit
        ``mean_only_accepts>0``, or inferred when accepts>0, mean moves, and
        fixed LL does not improve) while fixed LL does not improve.
    """
    fixed = _as_1d_finite("fixed_loglik_trace", fixed_loglik_trace)
    n_accepted = int(n_accepted_prior_steps)
    if n_accepted < 0:
        raise ValueError("n_accepted_prior_steps must be >= 0")
    if int(n_iter) < 1:
        raise ValueError("n_iter must be >= 1")

    mean_moved = prior_mean_trace_moves(
        prior_mean_trace,
        n_primary=int(n_primary),
        n_iter=int(n_iter),
        move_atol=move_atol,
    )
    flat = fixed_loglik_post_init_is_flat(fixed, atol=flat_atol)
    improved = fixed_loglik_improves(fixed, atol=flat_atol)

    # (1) mean path moves while frozen diagnostic LL is flat post-init.
    if mean_moved and flat:
        raise TwoTierFipcFixedMeasureRegressionError(
            "prior_mean_trace moves across iterations while fixed_loglik_trace "
            f"is flat after the first post-init value (flat_atol={flat_atol}); "
            f"fixed_loglik_trace={fixed.tolist()!r}"
        )

    # (2) mean-only accepts (explicit or inferred) without fixed-LL improvement.
    if mean_only_accepts is not None:
        mean_only = int(mean_only_accepts)
        if mean_only < 0:
            raise ValueError("mean_only_accepts must be >= 0")
        mean_only_path = mean_only > 0
    else:
        # Tip 0621169b had no mean-only counter: infer from accepts + mean move
        # + no fixed-LL improvement (writer mean-only recovery signature).
        mean_only_path = n_accepted > 0 and mean_moved and not improved

    if mean_only_path and n_accepted > 0 and not improved:
        raise TwoTierFipcFixedMeasureRegressionError(
            "n_accepted_prior_steps increased on a mean-only path while "
            "fixed_loglik_trace did not improve; "
            f"n_accepted_prior_steps={n_accepted}, "
            f"mean_only_accepts={mean_only_accepts!r}, "
            f"fixed_loglik_trace={fixed.tolist()!r}"
        )
