"""Relative DIF selection; Python validates/transports, Rust owns IC arithmetic.

Kang, T., Cohen, A. S., & Sung, H.-J. (2009). Model selection indices for
polytomous items. Applied Psychological Measurement, 33(7), 499–518.
https://doi.org/10.1177/0146621608327800 (pp. 500, 503; Bayesian DIC: p. 502,
eq. 4). See docs/dif_information_criteria.md for the pinned mirt audit,
prior restrictions and the separate Holm/GPU inference contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.machinery import ExtensionFileLoader
from importlib.util import module_from_spec, spec_from_loader
import math
import sys
import threading

import numpy as np

__all__ = [
    "DifExecutionReceipt",
    "DifFitEvidence",
    "DifInformationCriteriaComparison",
    "dif_fit_evidence",
    "compare_dif_information_criteria",
]
_NATIVE_NAME = "fast_mlsirm._dif_information_criteria_core"
_NATIVE_LOCK = threading.Lock()


def _native_core():
    """Load the secondary symbol from the same extension; never a fallback."""
    with _NATIVE_LOCK:
        cached = sys.modules.get(_NATIVE_NAME)
        if cached is not None:
            return cached
        from . import _core

        path = getattr(_core, "__file__", None)
        if not path:
            raise RuntimeError("compiled DIF information-criteria core is required")
        try:
            loader = ExtensionFileLoader(_NATIVE_NAME, path)
            spec = spec_from_loader(_NATIVE_NAME, loader)
            if spec is None:
                raise ImportError("cannot initialize DIF information-criteria core")
            module = module_from_spec(spec)
            loader.exec_module(module)
        except ImportError as exc:
            raise RuntimeError(
                "compiled DIF information-criteria core is required"
            ) from exc
        sys.modules[_NATIVE_NAME] = module
        return module


@dataclass(frozen=True)
class DifExecutionReceipt:
    """Copied native metadata, not authenticated proof or inferred stage coverage."""

    gpu_execution_used: bool | None = None
    backend: str | None = None
    device_name: str | None = None
    cpu_fallback_reason: str | None = None
    likelihood_stage: str = "unknown"
    m_step_stage: str = "unknown"


@dataclass(frozen=True)
class DifFitEvidence:
    """Caller-attested comparable observed-ML evidence; N counts persons."""

    loglik: float
    n_parameters: int
    n_persons: int
    converged: bool
    comparison_key: str
    likelihood_kind: str = "unpenalized_marginal_ml"
    execution: DifExecutionReceipt = field(default_factory=DifExecutionReceipt)


@dataclass(frozen=True)
class DifInformationCriteriaComparison:
    """Augmented−compact IC difference; no p-value, alpha or error-rate claim."""

    criterion: str
    compact_ic: float
    augmented_ic: float
    delta_ic: float
    prefer_augmented: bool
    arithmetic_backend: str = "rust_cpu_f64"
    inference: str = "relative_model_selection"
    compact_execution: DifExecutionReceipt = field(default_factory=DifExecutionReceipt)
    augmented_execution: DifExecutionReceipt = field(
        default_factory=DifExecutionReceipt
    )


def _integer(value, name: str, minimum: int) -> int:
    """Validate builtin/NumPy integer controls without lossy conversion."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    result = int(value)
    if result < minimum or result > np.iinfo(np.uintp).max:
        raise ValueError(f"{name} must be >= {minimum} and fit in usize")
    return result


def _validate_evidence(fit: DifFitEvidence) -> None:
    """Validate scalar evidence before native marshalling."""
    if type(fit) is not DifFitEvidence:
        raise ValueError("expected DifFitEvidence")
    if isinstance(fit.loglik, (bool, np.bool_)) or not isinstance(
        fit.loglik, (float, int, np.floating, np.integer)
    ):
        raise ValueError("loglik must be a finite real scalar")
    if not math.isfinite(fit.loglik):
        raise ValueError("loglik must be finite")
    _integer(fit.n_persons, "n_persons", 2)
    _integer(fit.n_parameters, "n_parameters", 0)
    if type(fit.converged) is not bool or not fit.converged:
        raise ValueError("DIF IC selection requires converged fits")
    if type(fit.comparison_key) is not str or not fit.comparison_key.strip():
        raise ValueError(
            "comparison_key must bind data, row order, estimator and integration"
        )
    if fit.likelihood_kind != "unpenalized_marginal_ml":
        raise ValueError(
            "ordinary ML IC requires unpenalized marginal ML, not MAP/prior/posterior objectives"
        )


def dif_fit_evidence(
    fit,
    *,
    n_persons: int,
    comparison_key: str,
    likelihood_kind: str | None = None,
) -> DifFitEvidence:
    """Copy final likelihood/native K without fitting or replacing MAP with ML.

    Untagged fits require an explicit ``likelihood_kind`` attestation. This
    prevents silently classifying an unknown penalized fit as maximized ML.
    The key attests matching responses, row order, estimator and integration.
    Missing stage receipts remain unknown; no requested device is used as proof.
    The method basis is Kang et al. (2009, pp. 500–503); references above.
    """
    for name in ("slope_prior_mu", "slope_prior_sd", "threshold_prior"):
        if getattr(fit, name, None) is not None:
            raise ValueError("MAP/prior fit cannot supply ordinary maximized-ML IC")
    recorded_kind = getattr(fit, "likelihood_kind", None)
    if (
        likelihood_kind is not None
        and recorded_kind is not None
        and likelihood_kind != recorded_kind
    ):
        raise ValueError("likelihood_kind contradicts the fit's recorded definition")
    kind = recorded_kind if likelihood_kind is None else likelihood_kind
    if kind is None:
        raise ValueError(
            "likelihood_kind must be explicit for an untagged fit; no ML/MAP assumption is made"
        )
    trace = np.asarray(getattr(fit, "loglik_trace", None))
    if (
        trace.ndim != 1
        or trace.size == 0
        or trace.dtype.kind not in "iuf"
        or not np.isfinite(trace[-1])
    ):
        raise ValueError(
            "loglik_trace must contain a finite final observed marginal loglik"
        )
    evidence = DifFitEvidence(
        loglik=float(trace[-1]),
        n_parameters=_integer(getattr(fit, "n_parameters", None), "n_parameters", 0),
        n_persons=_integer(n_persons, "n_persons", 2),
        converged=getattr(fit, "converged", None),
        comparison_key=comparison_key,
        likelihood_kind=kind,
        execution=DifExecutionReceipt(
            gpu_execution_used=getattr(fit, "gpu_execution_used", None),
            backend=getattr(fit, "gpu_backend", None),
            device_name=getattr(fit, "gpu_device_name", None),
            cpu_fallback_reason=getattr(fit, "cpu_fallback_reason", None),
        ),
    )
    _validate_evidence(evidence)
    return evidence


def compare_dif_information_criteria(
    compact: DifFitEvidence,
    augmented: DifFitEvidence,
    *,
    criterion: str,
) -> DifInformationCriteriaComparison:
    """Select only for a negative augmented−compact IC (exact tie: compact).

    Uses the existing Rust IC definitions. AICc/CAIC extend, not replicate, mirt
    DIF. DIC needs posterior deviances and is unsupported. IC is relative model
    selection (Kang et al., 2009, pp. 500, 503), not a Holm/FWER test.
    """
    _validate_evidence(compact)
    _validate_evidence(augmented)
    if type(criterion) is not str:
        raise ValueError("criterion must be an explicit string")

    def row(fit):
        return (
            float(fit.loglik),
            int(fit.n_parameters),
            int(fit.n_persons),
            fit.converged,
            fit.likelihood_kind,
            fit.comparison_key,
        )

    result = _native_core().compare_dif_information_criteria(
        row(compact), row(augmented), criterion
    )
    return DifInformationCriteriaComparison(
        **result,
        compact_execution=compact.execution,
        augmented_execution=augmented.execution,
    )
