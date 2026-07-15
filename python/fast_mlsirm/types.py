from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MLSIRMParams:
    theta: np.ndarray
    alpha: np.ndarray
    b: np.ndarray
    xi: np.ndarray
    zeta: np.ndarray
    tau: float

    @property
    def a(self) -> np.ndarray:
        return np.exp(self.alpha)

    @property
    def gamma(self) -> float:
        return float(np.exp(self.tau))

    def copy(self) -> "MLSIRMParams":
        return MLSIRMParams(
            theta=np.array(self.theta, copy=True),
            alpha=np.array(self.alpha, copy=True),
            b=np.array(self.b, copy=True),
            xi=np.array(self.xi, copy=True),
            zeta=np.array(self.zeta, copy=True),
            tau=float(self.tau),
        )


@dataclass
class SimulationData:
    Y: np.ndarray
    factor_id: np.ndarray
    truth: MLSIRMParams
    Phi: np.ndarray
    probabilities: np.ndarray
    config: Any


@dataclass
class FitResult:
    params: MLSIRMParams
    model: str
    optimizer: str
    backend: str
    rust_device: str
    objective: float
    loglik_trace: list[float]
    objective_trace: list[float]
    convergence_status: str
    n_iter: int
    # Marginal (MMLE) fits: population-structure estimates and posterior SDs.
    # Keys (present when applicable): "kind", "mu", "sigma" (multigroup),
    # "sigma_u", "u_eap", "icc" (multilevel), "theta_sd".
    population: dict[str, Any] | None = None
    # Marginal fits: information criteria (Kang, Cohen & Sung 2009) —
    # {"aic", "bic", "aicc", "sabic", "caic", "n_parameters", "n"}.
    ic: dict[str, Any] | None = None


@dataclass
class FitDiagnostics:
    itemfit: dict[str, np.ndarray]
    personfit: dict[str, np.ndarray]
    model_fit: dict[str, Any]
    factorfit: dict[str, np.ndarray] | None = None
    categoryfit: dict[str, np.ndarray] | None = None
    groupfit: dict[str, np.ndarray] | None = None
    clusterfit: dict[str, np.ndarray] | None = None
    group_itemfit: dict[str, np.ndarray] | None = None
    cluster_itemfit: dict[str, np.ndarray] | None = None


@dataclass
class DimensionalityDiagnostics:
    candidates: list[dict[str, float | str]]
    best: dict[str, float | str]


@dataclass
class RecoveryReport:
    summary: dict[str, float]
    metrics: dict[str, float]
