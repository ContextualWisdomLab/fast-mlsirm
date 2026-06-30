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
    objective: float
    loglik_trace: list[float]
    objective_trace: list[float]
    convergence_status: str
    n_iter: int


@dataclass
class FitDiagnostics:
    itemfit: dict[str, np.ndarray]
    personfit: dict[str, np.ndarray]
    model_fit: dict[str, float]


@dataclass
class RecoveryReport:
    summary: dict[str, float]
    metrics: dict[str, float]
