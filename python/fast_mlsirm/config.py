from __future__ import annotations

from dataclasses import dataclass

from .backend import normalize_backend, normalize_device


VALID_MODELS = {"MIRT", "MLS2PLM", "MLSRM", "ULS2PLM", "ULSRM"}
VALID_OPTIMIZERS = {"adam", "lbfgs", "adam_lbfgs"}
# Estimation methods. "jmle" (penalized joint MLE) is the legacy default; "mmle"
# (marginal MLE via EM) is robust to missing data. "em"/"bayes" are reserved
# for future milestones (the driver raises NotImplementedError for now).
VALID_ESTIMATORS = {"jmle", "mmle", "em", "bayes"}


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

    def normalized_model(self) -> str:
        return self.model.upper()

    def validate(self) -> None:
        model = self.normalized_model()
        if model not in VALID_MODELS:
            raise ValueError(f"model must be one of {sorted(VALID_MODELS)}")
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if self.optimizer not in VALID_OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {sorted(VALID_OPTIMIZERS)}")
        if self.estimator not in VALID_ESTIMATORS:
            raise ValueError(f"estimator must be one of {sorted(VALID_ESTIMATORS)}")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if self.n_restarts < 1:
            raise ValueError("n_restarts must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.init_gamma <= 0:
            raise ValueError("init_gamma must be > 0")
        import math
        if not math.isfinite(self.eps_distance) or self.eps_distance <= 0:
            raise ValueError("eps_distance must be finite and > 0")
        normalize_backend(self.backend)
        normalize_device(self.rust_device)
