# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Joint person bootstrap for the bifactor graded response model.

Resamples persons (stratified by group when multiple groups are present),
refits the Bock-Aitkin bifactor GRM on each replicate
(:func:`fast_mlsirm.bifactor_grm.fit_bifactor_grm` for one group,
:func:`fast_mlsirm.bifactor_multigroup.fit_bifactor_grm_multigroup` for
several), and reports empirical standard errors and percentile intervals
from the converged replicates. Replicate count, batch size, Monte Carlo
stopping ratio, and compute budget are caller arguments; this module defines
no study-specific defaults for them.

Implementation basis
--------------------
The replicate-number stopping rule is a sequential application of the accuracy
framework of Andrews and Buchinsky (2000, §§ 2–4): a finite-``B`` bootstrap
quantity is accurate when its percentage deviation from the ideal (``B`` → ∞)
bootstrap quantity is small. The caller-supplied ``mc_stopping_ratio`` plays
the role of their percentage-deviation bound ``pdb`` (expressed as a
fraction): after each batch, the percentile interval endpoints of every free
parameter are recomputed from all converged replicates so far, and the run
stops once the maximum endpoint movement relative to the interval half-width
falls below ``mc_stopping_ratio``. Because the ideal endpoints are unknown
mid-run, successive-batch endpoint movement is used as the observable proxy;
the compute budget always caps the run. Bias-corrected-and-accelerated (BCa)
intervals are out of scope.

References
----------
- Andrews, D. W. K., & Buchinsky, M. (2000). A three-step method for choosing
  the number of bootstrap repetitions. *Econometrica, 68*(1), 23–51.
  https://www.jstor.org/stable/2999474 (full text: Cowles Foundation Paper
  No. 1001, http://dido.econ.yale.edu/~dwka/pub/p1001.pdf; see eqs.
  (4.1)–(4.4) for the batch-size formulae and § 6 for the 95% interval
  simulations motivating the default ``ci_level``).
- Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E.,
  Bhaumik, D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., & Stover,
  A. (2007). Full-information item bifactor analysis of graded response
  data. *Applied Psychological Measurement, 31*(1), 4–19.
  https://doi.org/10.1177/0146621606289485
- Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor
  analysis. *Psychometrika, 57*(3), 423–436.
  https://doi.org/10.1007/BF02295430
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
import os
import time
from typing import Protocol

import numpy as np

from .bifactor_grm import fit_bifactor_grm
from .bifactor_multigroup import fit_bifactor_grm_multigroup
from .remote_exec import (
    RemoteJobDeliveryState,
    RemoteJobEnvelope,
    RemoteJobFamily,
    RemoteJobOutcome,
    RemoteRunManifest,
    payload_identity_sha256,
)

# Replicate error strings travel in remote result JSON; keep them bounded.
_MAX_REPLICATE_ERROR_CHARS = 512


def _require_gh_nodes(value: object, name: str) -> int:
    """Validate a Gauss-Hermite node count (any ``n >= 1``, no table cap).

    The Rust core generates any ``n >= 1`` rule on demand (Golub & Welsch,
    1969; issue #1929) and guards allocation overflow, so the Python layer
    only enforces the lower bound and integer-ness here.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be >= 1")
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be >= 1") from None
    if not np.isfinite(numeric) or numeric != np.floor(numeric):
        raise ValueError(f"{name} must be >= 1")
    intval = int(numeric)
    if intval < 1:
        raise ValueError(f"{name} must be >= 1")
    return intval


@dataclass
class BifactorBootstrapResult:
    """Summary and replicate storage of a joint person bootstrap run.

    replicate_ids and replicate_errors align with all completed flags;
    converged_replicate_ids identifies rows in the converged parameter arrays.
    Empty error strings denote successful fits. Defaults preserve construction
    compatibility; only the runner populates complete receipts. This is a
    provenance contract, not a statistical estimator. Python Software
    Foundation, Python dataclasses manual, field defaults and field order:
    https://docs.python.org/3/library/dataclasses.html .
    """

    n_requested: int
    n_replicates: int  # replicates actually completed (≤ n_requested)
    n_converged: int
    converged: np.ndarray  # bool per completed replicate, in replicate order
    stopped_early: bool  # True when the MC stopping rule fired
    ci_level: float
    n_groups: int
    replicate_a_general: np.ndarray  # (B_conv, n_items)
    replicate_a_specific: np.ndarray  # (B_conv, n_items)
    replicate_threshold: np.ndarray  # (B_conv, n_items, n_cat - 1)
    replicate_general_mean: np.ndarray  # (B_conv, n_groups)
    replicate_general_sd: np.ndarray  # (B_conv, n_groups)
    replicate_specific_sd: np.ndarray  # (B_conv, n_groups, n_specific)
    replicate_loglik: np.ndarray  # (B_conv,)
    se_a_general: np.ndarray
    se_a_specific: np.ndarray
    se_threshold: np.ndarray
    se_general_mean: np.ndarray
    se_general_sd: np.ndarray
    se_specific_sd: np.ndarray
    ci_lower_a_general: np.ndarray
    ci_upper_a_general: np.ndarray
    ci_lower_a_specific: np.ndarray
    ci_upper_a_specific: np.ndarray
    ci_lower_threshold: np.ndarray
    ci_upper_threshold: np.ndarray
    wall_clock_seconds: float
    throughput_replicates_per_second: float
    replicate_ids: tuple[int, ...] = ()
    converged_replicate_ids: tuple[int, ...] = ()
    replicate_errors: tuple[str, ...] = ()
    device: str = "cpu"
    rejected_replicate_ids: tuple[int, ...] = ()

    @property
    def n_rejected(self) -> int:
        """Replicates whose resample could not be fit; results, never redrawn."""
        return len(self.rejected_replicate_ids)


def _generate_bootstrap_indices(
    n_persons: int,
    group_ids: np.ndarray | None,
    n_groups: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate bootstrap sample indices, with stratification if multiple groups.

    Implementation basis: nonparametric iid person resampling within each
    known group (the resampling scheme to which Andrews and Buchinsky (2000,
    § 2) apply their replicate-number results for iid data; Andrews, D. W. K.,
    & Buchinsky, M. (2000). A three-step method for choosing the number of
    bootstrap repetitions. *Econometrica, 68*(1), 23–51.
    https://www.jstor.org/stable/2999474).
    """
    if group_ids is None or n_groups <= 1:
        return rng.integers(0, n_persons, size=n_persons, endpoint=False)

    sample_indices = np.empty(n_persons, dtype=np.int64)
    for g in range(n_groups):
        g_mask = np.flatnonzero(group_ids == g)
        if len(g_mask) == 0:
            continue
        g_sample = rng.choice(g_mask, size=len(g_mask), replace=True)
        sample_indices[g_mask] = g_sample
    return sample_indices


def _fit_single_replicate(
    rep_idx: int,
    responses: np.ndarray,
    specific_map: np.ndarray,
    n_cat: int,
    n_specific: int,
    group_ids: np.ndarray | None,
    n_groups: int,
    anchor_mask: np.ndarray | None,
    q_general: int,
    q_specific: int,
    max_iter: int,
    tol: float,
    n_starts: int,
    rep_seed: int,
    estimate_specific_vars: bool,
    device: str,
) -> tuple:
    """Execute one bootstrap resample and fit.

    The replicate seed derives deterministically from the caller-supplied
    ``base_seed`` and the replicate index, so CPU and GPU runs with the same
    ``base_seed`` draw identical resamples and identical start jitter and
    match replicate-by-replicate up to device precision.
    """
    rng = np.random.Generator(np.random.PCG64(rep_seed))
    n_persons = responses.shape[0]
    n_items = responses.shape[1]

    indices = _generate_bootstrap_indices(n_persons, group_ids, n_groups, rng)
    y_boot = responses[indices]
    g_boot = group_ids[indices] if group_ids is not None else None

    def _nan_result(converged: bool, err: str, rejected: bool):
        m1 = n_cat - 1
        return (
            rep_idx,
            converged,
            np.full(n_items, np.nan),
            np.full(n_items, np.nan),
            np.full((n_items, m1), np.nan),
            np.full(n_groups, np.nan),
            np.full(n_groups, np.nan),
            np.full((n_groups, n_specific), np.nan),
            float("nan"),
            err[:_MAX_REPLICATE_ERROR_CHARS],
            rejected,
        )

    try:
        if group_ids is None or n_groups <= 1:
            fit = fit_bifactor_grm(
                responses=y_boot,
                specific_map=specific_map,
                n_cat=n_cat,
                n_specific=n_specific,
                q_general=q_general,
                q_specific=q_specific,
                max_iter=max_iter,
                tol=tol,
                n_starts=n_starts,
                seed=rep_seed,
                device=device,
            )
            a_g = np.asarray(fit.a_general, dtype=np.float64)
            a_s = np.asarray(fit.a_specific, dtype=np.float64)
            thr = np.asarray(fit.threshold, dtype=np.float64).reshape(n_items, n_cat - 1)
            means = np.zeros(1)
            sds = np.ones(1)
            spec = np.ones((1, n_specific))
            ll = (
                float(fit.loglik_trace[-1]) if len(fit.loglik_trace) else float("nan")
            )
            return (
                rep_idx, bool(fit.converged), a_g, a_s, thr, means, sds, spec, ll,
                "" if fit.converged else f"fit not converged: {getattr(fit, 'termination_reason', 'unreported')}",
                False,
            )
        fit = fit_bifactor_grm_multigroup(
            responses=y_boot,
            group=g_boot,
            specific_map=specific_map,
            n_cat=n_cat,
            n_specific=n_specific,
            anchor_mask=anchor_mask,
            q_general=q_general,
            q_specific=q_specific,
            max_iter=max_iter,
            tol=tol,
            n_starts=n_starts,
            seed=rep_seed,
            estimate_specific_vars=estimate_specific_vars,
            device=device,
        )
        # Multigroup arrays are (n_groups, ...): the bootstrap resamples
        # persons, so per-replicate summaries keep the group axis.
        a_g = np.asarray(fit.a_general, dtype=np.float64)
        a_s = np.asarray(fit.a_specific, dtype=np.float64)
        thr = np.asarray(fit.threshold, dtype=np.float64)
        means = np.asarray(fit.general_mean, dtype=np.float64)
        sds = np.asarray(fit.general_sd, dtype=np.float64)
        spec = np.asarray(fit.specific_sd, dtype=np.float64)
        ll = float(fit.loglik_trace[-1]) if len(fit.loglik_trace) else float("nan")
        return (
            rep_idx, bool(fit.converged), a_g, a_s, thr, means, sds, spec, ll,
                "" if fit.converged else f"fit not converged: {getattr(fit, 'termination_reason', 'unreported')}",
            False,
        )
    except Exception as exc:  # noqa: BLE001 — replicate failure is data, reported via flags
        # A fit that raises (e.g. a resample missing a category) is a rejected
        # replicate: a result that is counted, never redrawn (#2001).
        return _nan_result(False, f"{type(exc).__name__}: {exc}", True)


def _json_floats(values: object) -> list:
    """Encode a float array as nested lists with NaN as ``None`` (JSON-safe)."""
    array = np.asarray(values, dtype=np.float64)
    encoded = array.astype(object)
    encoded[np.isnan(array)] = None
    return encoded.tolist()


def bootstrap_replicate_payload(
    responses: np.ndarray,
    specific_map: np.ndarray,
    n_cat: int,
    n_specific: int,
    group_ids: np.ndarray | None,
    n_groups: int,
    anchor_mask: np.ndarray | None,
    q_general: int,
    q_specific: int,
    max_iter: int,
    tol: float,
    n_starts: int,
    estimate_specific_vars: bool,
    device: str,
) -> dict[str, object]:
    """Serialize the replicate-invariant inputs of one bootstrap run.

    Every replicate of a run shares this payload; only the replicate index and
    its derived seed differ, and those travel in the job envelope.
    """
    return {
        "responses": _json_floats(responses),
        "specific_map": np.asarray(specific_map, dtype=np.int64).tolist(),
        "n_cat": int(n_cat),
        "n_specific": int(n_specific),
        "group_ids": None if group_ids is None else np.asarray(group_ids, dtype=np.int64).tolist(),
        "n_groups": int(n_groups),
        "anchor_mask": None if anchor_mask is None else np.asarray(anchor_mask, dtype=bool).tolist(),
        "q_general": int(q_general),
        "q_specific": int(q_specific),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "n_starts": int(n_starts),
        "estimate_specific_vars": bool(estimate_specific_vars),
        "device": str(device),
    }


def run_bootstrap_replicate_payload(
    payload: Mapping[str, object], rep_idx: int, rep_seed: int
) -> dict[str, object]:
    """Fit one replicate from a serialized run payload and encode the result.

    This is the worker-side entry point; it calls the same resample-and-fit
    body as the in-process path, so both produce identical replicates.
    """
    group_ids = payload["group_ids"]
    anchor_mask = payload["anchor_mask"]
    result = _fit_single_replicate(
        int(rep_idx),
        np.asarray(payload["responses"], dtype=np.float64),
        np.asarray(payload["specific_map"], dtype=np.int64),
        int(payload["n_cat"]),
        int(payload["n_specific"]),
        None if group_ids is None else np.asarray(group_ids, dtype=np.int64),
        int(payload["n_groups"]),
        None if anchor_mask is None else np.asarray(anchor_mask, dtype=bool),
        int(payload["q_general"]),
        int(payload["q_specific"]),
        int(payload["max_iter"]),
        float(payload["tol"]),
        int(payload["n_starts"]),
        int(rep_seed),
        bool(payload["estimate_specific_vars"]),
        str(payload["device"]),
    )
    return {
        "family": RemoteJobFamily.BIFACTOR_BOOTSTRAP_REPLICATE.value,
        "library_function": "fast_mlsirm.bifactor_bootstrap.run_bootstrap_replicate_payload",
        "replicate_id": result[0],
        "converged": result[1],
        "a_general": _json_floats(result[2]),
        "a_specific": _json_floats(result[3]),
        "threshold": _json_floats(result[4]),
        "general_mean": _json_floats(result[5]),
        "general_sd": _json_floats(result[6]),
        "specific_sd": _json_floats(result[7]),
        "loglik": _json_floats(result[8]),
        "error": result[9],
        "rejected": result[10],
    }


def _replicate_from_record(record: Mapping[str, object]) -> tuple:
    """Decode a worker replicate record into the in-process result tuple."""
    return (
        int(record["replicate_id"]),
        bool(record["converged"]),
        np.asarray(record["a_general"], dtype=np.float64),
        np.asarray(record["a_specific"], dtype=np.float64),
        np.asarray(record["threshold"], dtype=np.float64),
        np.asarray(record["general_mean"], dtype=np.float64),
        np.asarray(record["general_sd"], dtype=np.float64),
        np.asarray(record["specific_sd"], dtype=np.float64),
        float(np.asarray(record["loglik"], dtype=np.float64)),
        str(record["error"]),
        bool(record["rejected"]),
    )


class RemoteBatchExecutor(Protocol):
    """Anything that runs a batch of L4 envelopes, e.g. ``SubprocessExecutor``."""

    def run_batch(
        self,
        envelopes: Sequence[RemoteJobEnvelope],
        *,
        worker_manifest: RemoteRunManifest,
        payload: Mapping[str, object] | None = None,
    ) -> tuple[RemoteJobOutcome, ...]: ...


@dataclass(frozen=True)
class RemoteBootstrapBackend:
    """Run bootstrap replicates on an L4 executor instead of local threads.

    ``manifest`` is the cohort template; its ``payload_sha256`` is replaced by
    the identity of the run's serialized inputs. ``run_id`` names the run in
    every envelope, so a durable ledger can replay completed replicates.
    """

    executor: RemoteBatchExecutor
    manifest: RemoteRunManifest
    run_id: str


def _run_remote_batch(
    batch: list[tuple],
    backend: RemoteBootstrapBackend,
    manifest: RemoteRunManifest,
    payload: Mapping[str, object],
    base_seed: int,
    results: list,
) -> None:
    """Dispatch one batch of replicates and store their decoded results.

    Rejected replicates arrive as completed outcomes. A transport failure is
    not a statistical result, so the run stops instead of dropping the unit.
    """
    envelopes = []
    for task in batch:
        envelope = RemoteJobEnvelope(
            run_id=backend.run_id,
            family=RemoteJobFamily.BIFACTOR_BOOTSTRAP_REPLICATE,
            unit_index=task[0],
            base_seed=base_seed,
            payload_ref=manifest.payload_sha256,
            manifest=manifest,
        )
        if envelope.unit_seed() != task[13]:
            raise RuntimeError("remote seed derivation diverged from the bootstrap replicate seed")
        envelopes.append(envelope)
    outcomes = backend.executor.run_batch(envelopes, worker_manifest=manifest, payload=payload)
    failed = [o for o in outcomes if o.delivery_state is not RemoteJobDeliveryState.COMPLETED]
    if failed:
        raise RuntimeError(
            f"remote bootstrap replicate(s) {[o.unit_index for o in failed]} failed in transport: "
            f"{failed[0].error_message}"
        )
    for outcome in outcomes:
        record = _replicate_from_record(outcome.result)
        results[record[0]] = record


def _stack_monitor_vector(rows: list[np.ndarray]) -> np.ndarray:
    """Stack converged replicate summaries into one row-per-replicate matrix."""
    return np.concatenate([r[None, :] for r in rows], axis=0)


def _endpoint_movement(
    prev_lo: np.ndarray,
    prev_hi: np.ndarray,
    new_lo: np.ndarray,
    new_hi: np.ndarray,
) -> float:
    """Maximum endpoint movement relative to the new interval half-width.

    Entries with zero half-width (parameters that are constant across
    replicates, e.g. pinned reference-group moments) carry no Monte Carlo
    uncertainty and are excluded. This is the observable proxy for the
    percentage deviation of finite-``B`` interval endpoints from their ideal
    counterparts in Andrews and Buchinsky (2000, §§ 2–4).
    """
    half = (new_hi - new_lo) / 2.0
    live = half > 0
    if not np.any(live):
        return 0.0
    move = np.maximum(np.abs(new_lo[live] - prev_lo[live]),
                      np.abs(new_hi[live] - prev_hi[live])) / half[live]
    return float(np.max(move))


def _require_int(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def run_bifactor_bootstrap(
    responses: np.ndarray,
    specific_map: np.ndarray,
    n_cat: int,
    n_specific: int,
    n_replicates: int,
    batch_size: int,
    mc_stopping_ratio: float,
    compute_budget_seconds: float,
    q_general: int,
    q_specific: int,
    *,
    base_seed: int,
    ci_level: float,
    max_iter: int,
    n_starts: int,
    tol: float,
    group_ids: np.ndarray | None = None,
    n_groups: int = 1,
    anchor_mask: np.ndarray | None = None,
    n_jobs: int = -1,
    device: str = "cpu",
    estimate_specific_vars: bool = False,
    remote_backend: RemoteBootstrapBackend | None = None,
) -> BifactorBootstrapResult:
    """Run joint person bootstrap replications with parallel workers.

    Parameters:
        responses: Persons x items array of response categories.
        specific_map: Length-``n_items`` array with ``-1`` for general-only
            items and ``0..n_specific-1`` otherwise.
        n_cat: Number of response categories.
        n_specific: Number of specific factors.
        n_replicates: Total bootstrap replicates requested (caller-supplied;
            no study-specific default is defined by this module).
        batch_size: Replicates per batch; the stopping rule and the compute
            budget are evaluated at batch boundaries.
        mc_stopping_ratio: Bound on the Monte Carlo error of the percentile
            interval endpoints relative to the interval half-width, in the
            role of the percentage-deviation bound ``pdb`` of Andrews and
            Buchinsky (2000, §§ 2–4). Must satisfy ``0 <= ratio < 1``;
            ``0`` disables early stopping (the run completes all requested
            replicates within budget).
        compute_budget_seconds: Wall-clock budget; batch execution stops when
            the elapsed time reaches this bound.
        q_general/q_specific: Required Gauss-Hermite node counts (any integer
            ``>= 1``; generated on demand via Golub & Welsch, 1969 — no
            fixed-table cap, issue #1929); no default is offered.
        group_ids: Optional 1-D group membership indices (``None`` selects
            the single-group estimator).
        n_groups: Positive integer number of groups. With ``group_ids``, every
            declared label ``0..n_groups-1`` must occur. Without ``group_ids``,
            only ``n_groups=1`` is valid. Numeric labels must be finite exact
            integers; conversion must not silently change stratum membership.
        anchor_mask: Optional multigroup anchor mask (``None`` = all common).
        n_jobs: Number of parallel workers (-1 for all logical cores).
        base_seed: Master seed for deterministic replication. Required,
            keyword-only caller argument (ADR-0028, #1963): a stochastic
            routine must not ship a default seed.
        device: 'cpu', 'gpu', or 'auto' execution device.
        ci_level: Nominal level of the reported percentile intervals and of
            the endpoints monitored by the stopping rule (0 < level < 1).
            Required, keyword-only (ADR-0028, #1963): a decision-threshold
            cutoff with no cited source in this repository.
        max_iter: Max EM iterations per replicate. Required, keyword-only
            (ADR-0028, #1963): an iteration/convergence precision control
            with no documented convergence-criterion source.
        tol: Convergence tolerance. Required, keyword-only (ADR-0028, #1963):
            same reasoning as ``max_iter``.
        n_starts: EM starts per replicate (deterministic from the replicate
            seed). Required, keyword-only (ADR-0028, #1963): a
            replicate/bootstrap/draw count with no Monte-Carlo-error
            justification on file.
        estimate_specific_vars: Multigroup focal specific-variance estimation.
        remote_backend: Optional L4 executor for the replicates (#2001). When
            set, each replicate becomes one ``bifactor_bootstrap_replicate``
            job envelope and ``n_jobs`` is ignored (the executor owns
            concurrency). Remote replicates use the same resample-and-fit body
            and seed as local ones, so results match replicate by replicate.
            A transport failure raises ``RuntimeError``; with a durable ledger
            a rerun replays completed replicates instead of redrawing them.

    Returns:
        BifactorBootstrapResult with replicate matrices, empirical SEs,
        percentile intervals, per-replicate convergence flags, and timing.
        Replicates that fail to converge (or raise) are reported in
        ``converged`` and excluded from all summary statistics; failed
        replicates are never substituted or imputed. A replicate whose fit
        raises (for example a resample that misses a response category) is
        listed in ``rejected_replicate_ids`` with its reason in
        ``replicate_errors``. It is a result, not a failure: it is never
        redrawn, because redrawing until success biases the bootstrap sample,
        and callers must report ``n_rejected`` next to the summaries rather
        than average over it. When no replicate
        converges, a ``RuntimeError`` carrying the first replicate's error
        is raised instead of returning empty summaries. Its replicate_ids,
        converged_replicate_ids, replicate_errors and converged attributes
        preserve the same completed-replicate receipt as a successful result.
        These attributes are an implementation reporting contract; RuntimeError
        remains the exception type (Python Software Foundation, Built-in
        Exceptions manual: https://docs.python.org/3/library/exceptions.html ).

    References:
        Andrews, D. W. K., & Buchinsky, M. (2000). A three-step method for
        choosing the number of bootstrap repetitions. *Econometrica, 68*(1),
        23–51. https://www.jstor.org/stable/2999474

        NumPy Developers. NumPy reference manual, ``ndarray.astype``,
        Parameters (``casting``) and Examples:
        https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html
        The default unsafe cast can truncate fractional labels. Validate
        labels before conversion. ``numpy.empty``, Notes:
        https://numpy.org/doc/stable/reference/generated/numpy.empty.html
        Every element must be written before reading; requiring all labels
        to belong to declared strata guarantees the sampler fills every row.
        These checks enforce the existing stratum contract, not a choice of
        scientifically appropriate strata.
    """
    n_replicates = _require_int(n_replicates, "n_replicates", 1)
    batch_size = _require_int(batch_size, "batch_size", 1)
    n_starts = _require_int(n_starts, "n_starts", 1)
    max_iter = _require_int(max_iter, "max_iter", 1)
    if (
        isinstance(mc_stopping_ratio, bool)
        or not isinstance(mc_stopping_ratio, (float, int))
        or not math.isfinite(mc_stopping_ratio)
        or not 0.0 <= mc_stopping_ratio < 1.0
    ):
        raise ValueError(
            "mc_stopping_ratio must satisfy 0 <= ratio < 1 "
            f"(0 disables early stopping), got {mc_stopping_ratio!r}"
        )
    if (
        isinstance(compute_budget_seconds, bool)
        or not isinstance(compute_budget_seconds, (float, int))
        or not math.isfinite(compute_budget_seconds)
        or compute_budget_seconds <= 0
    ):
        raise ValueError(
            f"compute_budget_seconds must be positive and finite, got {compute_budget_seconds!r}"
        )
    if (
        isinstance(ci_level, bool)
        or not isinstance(ci_level, (float, int))
        or not math.isfinite(ci_level)
        or not 0.0 < ci_level < 1.0
    ):
        raise ValueError(f"ci_level must satisfy 0 < level < 1, got {ci_level!r}")
    if not isinstance(device, str) or device.strip().lower() not in ("cpu", "gpu", "auto"):
        raise ValueError(f"device must be one of 'cpu', 'gpu', 'auto'; got {device!r}")
    device = device.strip().lower()
    q_general = _require_gh_nodes(q_general, "q_general")
    q_specific = _require_gh_nodes(q_specific, "q_specific")
    if not isinstance(tol, (float, int)) or not math.isfinite(tol) or tol <= 0:
        raise ValueError(f"tol must be finite and positive, got {tol!r}")
    if not isinstance(estimate_specific_vars, bool):
        raise ValueError("estimate_specific_vars must be a bool")

    y_arr = np.asarray(responses, dtype=np.float64)
    if y_arr.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y_arr.shape
    smap_arr = np.asarray(specific_map)
    n_groups = _require_int(n_groups, "n_groups", 1)
    if n_persons < 1 or n_groups > n_persons:
        raise ValueError("n_groups requires at least one person in every declared group")
    g_arr = np.asarray(group_ids) if group_ids is not None else None
    if g_arr is None:
        if n_groups != 1:
            raise ValueError("group_ids is required when n_groups > 1")
    else:
        if g_arr.ndim != 1 or g_arr.size != n_persons:
            raise ValueError("group_ids must have length n_persons")
        if g_arr.dtype.kind not in ("i", "u", "f"):
            raise ValueError("group_ids must contain numeric integer labels")
        if g_arr.dtype.kind == "f" and (
            not bool(np.isfinite(g_arr).all())
            or bool((g_arr != np.floor(g_arr)).any())
        ):
            raise ValueError("group_ids must contain finite integer labels")
        if bool((g_arr < 0).any()) or bool((g_arr >= n_groups).any()):
            raise ValueError("group_ids must be in 0..n_groups-1")
        g_arr = g_arr.astype(np.int64, copy=False)
        if np.unique(g_arr).size != n_groups:
            raise ValueError("group_ids must represent every declared group")
    anchor_arr = np.asarray(anchor_mask, dtype=bool) if anchor_mask is not None else None

    start_time = time.perf_counter()
    if n_jobs <= 0:
        n_jobs = max(1, os.cpu_count() or 1)

    alpha = 1.0 - float(ci_level)
    lo_q, hi_q = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)

    # Prepare replicate tasks with deterministic seeds.
    tasks = []
    for b in range(n_replicates):
        # 64-bit golden-ratio step for uncorrelated replicate seeds.
        rep_seed = int((base_seed + b * 0x9E37_79B9_7F4A_7C15) & 0xFFFF_FFFF_FFFF_FFFF)
        tasks.append((
            b, y_arr, smap_arr, n_cat, n_specific, g_arr, n_groups, anchor_arr,
            q_general, q_specific, max_iter, float(tol), n_starts, rep_seed,
            estimate_specific_vars, device,
        ))

    results: list = [None] * n_replicates
    if remote_backend is not None:
        remote_payload = bootstrap_replicate_payload(
            y_arr, smap_arr, n_cat, n_specific, g_arr, n_groups, anchor_arr,
            q_general, q_specific, max_iter, float(tol), n_starts,
            estimate_specific_vars, device,
        )
        remote_manifest = replace(
            remote_backend.manifest,
            payload_sha256=payload_identity_sha256(remote_payload),
        )
    batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
    completed_reps = 0
    stopped_early = False
    prev_lo: np.ndarray | None = None
    prev_hi: np.ndarray | None = None
    batches_completed = 0

    for batch_pos, batch in enumerate(batches):
        if time.perf_counter() - start_time >= compute_budget_seconds:
            break
        if remote_backend is not None:
            _run_remote_batch(
                batch, remote_backend, remote_manifest, remote_payload, base_seed, results
            )
        elif n_jobs == 1 or len(batch) == 1:
            for t in batch:
                res = _fit_single_replicate(*t)
                results[res[0]] = res
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
                futures = [executor.submit(_fit_single_replicate, *t) for t in batch]
                for f in concurrent.futures.as_completed(futures):
                    res = f.result()
                    results[res[0]] = res
        completed_reps += len(batch)
        batches_completed += 1

        # Monte Carlo stopping check at batch boundaries (needs two endpoint
        # estimates to measure movement, so it can only fire from the second
        # completed batch on, and never on the final batch).
        last_batch = batch_pos == len(batches) - 1
        if mc_stopping_ratio > 0 and batches_completed >= 2 and not last_batch:
            conv_rows = [
                np.concatenate([
                    np.ravel(results[t[0]][2]), np.ravel(results[t[0]][3]),
                    np.ravel(results[t[0]][4]), np.ravel(results[t[0]][5]),
                    np.ravel(results[t[0]][6]), np.ravel(results[t[0]][7]),
                ])
                for t in tasks[:completed_reps]
                if results[t[0]] is not None and results[t[0]][1]
            ]
            if conv_rows:
                stacked = _stack_monitor_vector(conv_rows)
                new_lo = np.percentile(stacked, lo_q, axis=0)
                new_hi = np.percentile(stacked, hi_q, axis=0)
                if prev_lo is not None and prev_hi is not None:
                    if _endpoint_movement(prev_lo, prev_hi, new_lo, new_hi) < mc_stopping_ratio:
                        stopped_early = True
                        break
                prev_lo, prev_hi = new_lo, new_hi

    done = [r for r in results[:completed_reps] if r is not None]
    flags = np.array([r[1] for r in done], dtype=bool)
    conv = [r for r in done if r[1]]
    n_conv = len(conv)
    if n_conv == 0:
        first_err = done[0][9] if done else "no replicate completed"
        error = RuntimeError(
            f"joint person bootstrap: 0/{completed_reps} replicates converged; "
            f"first replicate error: {first_err}"
        )
        error.replicate_ids = tuple(r[0] for r in done)
        error.converged_replicate_ids = ()
        error.replicate_errors = tuple(r[9] for r in done)
        error.rejected_replicate_ids = tuple(r[0] for r in done if r[10])
        error.converged = flags
        raise error

    m1 = n_cat - 1
    if group_ids is None or n_groups <= 1:
        eff_groups = 1
        slopes_mat = np.stack([r[2] for r in conv], axis=0)
        spec_mat = np.stack([r[3] for r in conv], axis=0)
        thresh_mat = np.stack([r[4] for r in conv], axis=0).reshape(n_conv, n_items, m1)
        means_mat = np.stack([r[5] for r in conv], axis=0).reshape(n_conv, 1)
        sds_mat = np.stack([r[6] for r in conv], axis=0).reshape(n_conv, 1)
        sspec_mat = np.stack([r[7] for r in conv], axis=0).reshape(n_conv, 1, -1)
    else:
        eff_groups = n_groups
        slopes_mat = np.stack([r[2] for r in conv], axis=0).reshape(n_conv, n_groups, n_items)
        spec_mat = np.stack([r[3] for r in conv], axis=0).reshape(n_conv, n_groups, n_items)
        thresh_mat = np.stack([r[4] for r in conv], axis=0).reshape(
            n_conv, n_groups, n_items, m1
        )
        means_mat = np.stack([r[5] for r in conv], axis=0).reshape(n_conv, n_groups)
        sds_mat = np.stack([r[6] for r in conv], axis=0).reshape(n_conv, n_groups)
        sspec_mat = np.stack([r[7] for r in conv], axis=0).reshape(
            n_conv, n_groups, -1
        )
    # Collapse the group axis for the single-group path so summaries keep the
    # item-major shapes callers expect.
    if eff_groups == 1 and (group_ids is None or n_groups <= 1):
        slopes_flat = slopes_mat
        spec_flat = spec_mat
        thresh_flat = thresh_mat
    else:
        slopes_flat = slopes_mat.reshape(n_conv, -1)
        spec_flat = spec_mat.reshape(n_conv, -1)
        thresh_flat = thresh_mat.reshape(n_conv, -1)
    loglik_vec = np.array([r[8] for r in conv], dtype=np.float64)

    def _summarize(mat: np.ndarray):
        if mat.shape[0] > 1:
            return (
                np.std(mat, axis=0, ddof=1),
                np.percentile(mat, lo_q, axis=0),
                np.percentile(mat, hi_q, axis=0),
            )
        nan = np.full(mat.shape[1:], np.nan)
        return nan, nan, nan

    se_ag, lo_ag, hi_ag = _summarize(slopes_flat)
    se_as, lo_as, hi_as = _summarize(spec_flat)
    se_th, lo_th, hi_th = _summarize(thresh_flat)
    se_mn, _, _ = _summarize(means_mat.reshape(n_conv, -1))
    se_sd, _, _ = _summarize(sds_mat.reshape(n_conv, -1))
    se_ss, _, _ = _summarize(sspec_mat.reshape(n_conv, -1))

    # Restore structured shapes for item-major summaries.
    def _shape(mat: np.ndarray, shape: tuple) -> np.ndarray:
        return mat.reshape(shape)

    if eff_groups == 1 and (group_ids is None or n_groups <= 1):
        se_ag_s, lo_ag_s, hi_ag_s = se_ag, lo_ag, hi_ag
        se_as_s, lo_as_s, hi_as_s = se_as, lo_as, hi_as
        se_th_s, lo_th_s, hi_th_s = (
            _shape(se_th, (n_items, m1)),
            _shape(lo_th, (n_items, m1)),
            _shape(hi_th, (n_items, m1)),
        )
        rep_ag, rep_as = slopes_mat, spec_mat
        rep_th = thresh_mat
    else:
        se_ag_s, lo_ag_s, hi_ag_s = (
            _shape(se_ag, (n_groups, n_items)),
            _shape(lo_ag, (n_groups, n_items)),
            _shape(hi_ag, (n_groups, n_items)),
        )
        se_as_s, lo_as_s, hi_as_s = (
            _shape(se_as, (n_groups, n_items)),
            _shape(lo_as, (n_groups, n_items)),
            _shape(hi_as, (n_groups, n_items)),
        )
        se_th_s, lo_th_s, hi_th_s = (
            _shape(se_th, (n_groups, n_items, m1)),
            _shape(lo_th, (n_groups, n_items, m1)),
            _shape(hi_th, (n_groups, n_items, m1)),
        )
        rep_ag = slopes_mat
        rep_as = spec_mat
        rep_th = thresh_mat
    se_mn_s = se_mn.reshape(eff_groups)
    se_sd_s = se_sd.reshape(eff_groups)
    se_ss_s = se_ss.reshape(eff_groups, -1)

    elapsed = time.perf_counter() - start_time
    throughput = completed_reps / elapsed if elapsed > 0 else 0.0

    return BifactorBootstrapResult(
        n_requested=n_replicates,
        n_replicates=completed_reps,
        n_converged=n_conv,
        converged=flags,
        stopped_early=stopped_early,
        ci_level=float(ci_level),
        n_groups=eff_groups,
        replicate_a_general=rep_ag,
        replicate_a_specific=rep_as,
        replicate_threshold=rep_th,
        replicate_general_mean=means_mat,
        replicate_general_sd=sds_mat,
        replicate_specific_sd=sspec_mat,
        replicate_loglik=loglik_vec,
        se_a_general=se_ag_s,
        se_a_specific=se_as_s,
        se_threshold=se_th_s,
        se_general_mean=se_mn_s,
        se_general_sd=se_sd_s,
        se_specific_sd=se_ss_s,
        ci_lower_a_general=lo_ag_s,
        ci_upper_a_general=hi_ag_s,
        ci_lower_a_specific=lo_as_s,
        ci_upper_a_specific=hi_as_s,
        ci_lower_threshold=lo_th_s,
        ci_upper_threshold=hi_th_s,
        wall_clock_seconds=elapsed,
        throughput_replicates_per_second=throughput,
        replicate_ids=tuple(r[0] for r in done),
        converged_replicate_ids=tuple(r[0] for r in conv),
        replicate_errors=tuple(r[9] for r in done),
        device=device,
        rejected_replicate_ids=tuple(r[0] for r in done if r[10]),
    )
