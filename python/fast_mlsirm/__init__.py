from importlib.metadata import PackageNotFoundError, version

from .config import FitConfig as FitConfig
from .config import MLS2PLMConfig as MLS2PLMConfig
from .config import PenaltyConfig as PenaltyConfig
from .diagnostics import align_latent_space as align_latent_space
from .diagnostics import \
    dimensionality_diagnostics as dimensionality_diagnostics
from .diagnostics import fit_diagnostics as fit_diagnostics
from .diagnostics import predict_proba as predict_proba
from .diagnostics import recovery_report as recovery_report
from .diagnostics import \
    response_process_dimensionality_diagnostics as \
    response_process_dimensionality_diagnostics
from .diagnostics import \
    response_process_fit_diagnostics as response_process_fit_diagnostics
from .fit import fit as fit
from .report import render_diagnostics_report as render_diagnostics_report
from .simulation import simulate as simulate
from .types import DimensionalityDiagnostics as DimensionalityDiagnostics
from .types import FitDiagnostics as FitDiagnostics
from .types import FitResult as FitResult
from .types import MLSIRMParams as MLSIRMParams
from .types import RecoveryReport as RecoveryReport
from .types import SimulationData as SimulationData

try:
    __version__ = version("fast-mlsirm")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "DimensionalityDiagnostics",
    "FitConfig",
    "FitDiagnostics",
    "FitResult",
    "MLS2PLMConfig",
    "MLSIRMParams",
    "PenaltyConfig",
    "RecoveryReport",
    "SimulationData",
    "align_latent_space",
    "dimensionality_diagnostics",
    "fit",
    "fit_diagnostics",
    "predict_proba",
    "recovery_report",
    "response_process_dimensionality_diagnostics",
    "response_process_fit_diagnostics",
    "render_diagnostics_report",
    "simulate",
]
