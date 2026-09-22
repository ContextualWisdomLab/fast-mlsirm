"""Public package surface for fast-mlsirm."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version

from . import _legacy_init as _legacy_init
from . import cat as _cat
from . import cdm as _cdm
from . import dif as _dif
from . import exposure as _exposure
from . import fitstats as _fitstats
from . import inference as _inference
from . import polytomous as _polytomous
from . import reliability as _reliability
from . import scaling as _scaling
from . import serving as _serving
from . import validation as _validation
from ._cat_administration_resource_safety import (
    install as _install_cat_administration_resource_safety,
)
from ._cdm_response_safety import install as _install_cdm_response_safety
from ._dif_control_safety import install as _install_dif_control_safety
from ._exposure_array_safety import install as _install_exposure_array_safety
from ._exposure_flexilevel_safety import install as _install_exposure_flexilevel_safety
from ._fit_public import fit as _public_fit
from ._fitstats_control_safety import install as _install_fitstats_control_safety
from ._fleiss_control_safety import install as _install_fleiss_control_safety
from ._icc_control_safety import install as _install_icc_control_safety
from ._inference_admission_safety import install as _install_inference_admission_safety
from ._polytomous_prediction_admission import (
    install as _install_polytomous_prediction_admission,
)
from ._scaling_control_safety import install as _install_scaling_control_safety
from ._serving_export_safety import install as _install_serving_export_safety
from .interaction_map import ResidualInteractionMap as ResidualInteractionMap
from .interaction_map import residual_interaction_map as residual_interaction_map
from .sampling_design import (
    ACHIEVED_PROPORTION_SCHEMA_VERSION as ACHIEVED_PROPORTION_SCHEMA_VERSION,
)
from .sampling_design import (
    AchievedProportion as AchievedProportion,
)
from .sampling_design import (
    ProportionSamplingDesign as ProportionSamplingDesign,
)
from .sampling_design import SAMPLING_DESIGN_SCHEMA_VERSION as SAMPLING_DESIGN_SCHEMA_VERSION
from .sampling_design import SamplingStratum as SamplingStratum
from .sampling_design import (
    finite_population_achieved_proportion as finite_population_achieved_proportion,
)
from .sampling_design import (
    finite_population_proportion_design as finite_population_proportion_design,
)
from .bifactor_recursion import (
    bifactor_lord_wingersky as bifactor_lord_wingersky,
    direct_enumeration_bifactor as direct_enumeration_bifactor,
    enumerate_bifactor_direct as enumerate_bifactor_direct,
    enumerate_bifactor_lord_wingersky as enumerate_bifactor_lord_wingersky,
)
from .bifactor_bootstrap import (
    BifactorBootstrapResult as BifactorBootstrapResult,
    run_bifactor_bootstrap as run_bifactor_bootstrap,
)

# Harden historical public adapters before copying legacy exports. These
# wrappers validate and normalize semantic controls/evidence only; result
# arithmetic remains in the existing Rust-backed implementations.
_install_cat_administration_resource_safety(_cat)
_install_exposure_array_safety(_exposure)
_install_exposure_flexilevel_safety(_exposure)
_install_fitstats_control_safety(_fitstats)
_install_cdm_response_safety(_cdm)
_install_icc_control_safety(_reliability)

_install_dif_control_safety(_dif)
for _dif_name in (
    "detect_dif_logistic",
    "detect_dif_mantel_haenszel_purified",
    "detect_dif_logistic_purified",
    "detect_dif_mantel_haenszel",
    "detect_dif_mantel_smd",
    "detect_dif_gmh",
    "detect_dif_breslow_day",
):
    if hasattr(_legacy_init, _dif_name):
        setattr(_legacy_init, _dif_name, getattr(_dif, _dif_name))
