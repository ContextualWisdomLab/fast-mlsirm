from __future__ import annotations

import math
from dataclasses import dataclass

from .backend import normalize_backend, normalize_device


VALID_MODELS = {"MIRT", "MLS2PLM", "MLSRM", "ULS2PLM", "ULSRM", "BIFAC2PLM"}
VALID_OPTIMIZERS = {"adam", "lbfgs", "adam_lbfgs"}
# Estimation methods. "jmle" (penalized joint MLE) is the legacy default; "mmle"
# (marginal MLE via EM) is robust to missing data. "em"/"bayes" are reserved
# for future milestones (the driver raises NotImplementedError for now).
VALID_ESTIMATORS = {"jmle", "mmle", "em", "bayes"}

# Hard upper bounds on caller-supplied sizes, to reject sparse/oversized
# configurations that would force huge allocations before any real work
# (defense against memory-exhaustion DoS from untrusted fit settings).
# latent_dim: the joint grid is q_xi**latent_dim and Halton QMC supports
# only len(_HALTON_PRIMES) = 6 axes, so 8 is already generous.
MAX_LATENT_DIM = 8
MAX_XI_POINTS = 1_000_000
MAX_MAX_ITER = 100_000
MAX_RESTARTS = 1_000
MAX_M_STEPS = 1_000


@dataclass(frozen=True)
class MLS2PLMConfig:
    n_persons: int = 500
    n_dims: int = 2
    items_per_dim: int = 8
    latent_dim: int = 2
    phi: float = 0.3
    gamma: float = 1.5
    seed: int = 1
    dtype: str = "float64"

    @property
    def n_items(self) -> int:
        return self.n_dims * self.items_per_dim

    def validate(self) -> None:
        if self.n_persons < 1:
            raise ValueError("n_persons must be >= 1")
        if self.n_dims < 1:
            raise ValueError("n_dims must be >= 1")
        if self.items_per_dim < 1:
            raise ValueError("items_per_dim must be >= 1")
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if not (-1.0 / max(self.n_dims - 1, 1) < self.phi < 1.0):
            raise ValueError("phi must produce a positive-definite equicorrelation matrix")
        if self.gamma < 0:
            raise ValueError("gamma must be >= 0")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")


@dataclass(frozen=True)
class PenaltyConfig:
    lambda_theta: float = 0.01
    lambda_xi: float = 0.01
    lambda_zeta: float = 0.01
    lambda_b: float = 0.001
    lambda_alpha: float = 0.001
    lambda_tau: float = 0.001
    mu_alpha: float = 0.0
    mu_tau: float = 0.0


@dataclass(frozen=True)
class FitConfig:
    model: str = "MLS2PLM"
    latent_dim: int = 2
    optimizer: str = "adam_lbfgs"
    estimator: str = "jmle"
    max_iter: int = 1000
    n_restarts: int = 5
    learning_rate: float = 0.01
    seed: int = 1
    eps_distance: float = 1e-8
    init_gamma: float = 1.0
    tolerance: float = 1e-6
    gradient_clip: float | None = 100.0
    lbfgs_history: int = 10
    verbose: int = 0
    # Rust is the primary numeric path: "auto" resolves to the compiled
    # ``fast_mlsirm._core`` (Rust/PyO3) kernel when available and transparently
    # falls back to the pure-numpy reference implementation otherwise.
    backend: str = "auto"
    # Device for the Rust backend: "cpu", "gpu", or "auto". A sub-option of the
    # rust backend, not a separate compute-backend axis. "auto" (default) uses
    # the wgpu GPGPU kernels when a GPU is available and otherwise falls back to
    # the identical CPU path. Ignored when backend == "numpy".
    rust_device: str = "auto"
    penalty: PenaltyConfig = PenaltyConfig()
    # Marginal (MMLE) estimator quadrature: Gauss-Hermite nodes per trait
    # dimension, per latent-space axis (tensor grid of q_xi**latent_dim), and
    # for the multilevel random intercept. Supported sizes: 7/11/15/21/31/41.
    q_theta: int = 21
    q_xi: int = 11
    q_u: int = 15
    # Fisher-preconditioned ascent steps per item per M-step (marginal EM).
    m_steps: int = 4
    # Latent-space integration rule for the marginal estimator: "gh" (tensor
    # Gauss-Hermite, q_xi per axis), "qmc" (Halton QMC-EM, Jank 2005) or "mc"
    # (seeded Monte Carlo EM, Wei & Tanner 1990).
    xi_rule: str = "gh"
    # Point count for the qmc/mc rules; xi_seed is the Halton random shift /
    # Monte Carlo seed (deterministic, mirrored across backends).
    xi_points: int = 256
    xi_seed: int = 0
    # Zero-inflated mixture (marginal estimator): a structural-zero latent
    # class produces all-zero patterns with probability pi (estimated by EM);
    # cf. the ZI count-model guidance of Perumean-Chaney et al. (2013).
    zero_inflation: bool = False

    def normalized_model(self) -> str:
        return self.model.upper()

    def validate(self) -> None:
        model = self.normalized_model()
        if model not in VALID_MODELS:
            raise ValueError(f"model must be one of {sorted(VALID_MODELS)}")
        if not (1 <= self.latent_dim <= MAX_LATENT_DIM):
            raise ValueError(f"latent_dim must be >= 1 and <= {MAX_LATENT_DIM}")
        if self.optimizer not in VALID_OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {sorted(VALID_OPTIMIZERS)}")
        if self.estimator not in VALID_ESTIMATORS:
            raise ValueError(f"estimator must be one of {sorted(VALID_ESTIMATORS)}")
        if not (1 <= self.max_iter <= MAX_MAX_ITER):
            raise ValueError(f"max_iter must be >= 1 and <= {MAX_MAX_ITER}")
        if not (1 <= self.n_restarts <= MAX_RESTARTS):
            raise ValueError(f"n_restarts must be >= 1 and <= {MAX_RESTARTS}")
        # non-finite floats (NaN/Inf) slip past bare `<= 0` comparisons
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0 and finite")
        if not math.isfinite(self.init_gamma) or self.init_gamma <= 0:
            raise ValueError("init_gamma must be > 0 and finite")
        if not math.isfinite(self.eps_distance) or self.eps_distance <= 0:
            raise ValueError("eps_distance must be > 0 and finite")
        if not math.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be > 0 and finite")
        if self.gradient_clip is not None and (
            not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0
        ):
            raise ValueError("gradient_clip must be > 0 and finite, or None")
        supported_q = {7, 11, 15, 21, 31, 41}
        for name in ("q_theta", "q_xi", "q_u"):
            if getattr(self, name) not in supported_q:
                raise ValueError(f"{name} must be one of {sorted(supported_q)}")
        if not (1 <= self.m_steps <= MAX_M_STEPS):
            raise ValueError(f"m_steps must be >= 1 and <= {MAX_M_STEPS}")
        if self.xi_rule.lower() not in {"gh", "qmc", "halton", "mc", "montecarlo", "monte-carlo"}:
            raise ValueError("xi_rule must be one of ['gh', 'qmc', 'mc']")
        if not (1 <= self.xi_points <= MAX_XI_POINTS):
            raise ValueError(f"xi_points must be >= 1 and <= {MAX_XI_POINTS}")
        normalize_backend(self.backend)
        normalize_device(self.rust_device)
