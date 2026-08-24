"""Public contextual-membership contracts and crossed ``u_h`` estimation.

Contracts remain the sealed design layer. ``estimate_crossed_person_effects``
is the Rust-owned MAP estimator of multiple-membership / crossed person
effects (Fox & Glas, 2001; Browne, Goldstein, & Rasbash, 2001).
"""

from .contracts import (
    ContextMembership,
    ContextMembershipDesign,
    LongitudinalDesign,
    LongitudinalStateKind,
    LongitudinalStateSpec,
    MultilevelContractError,
    TemporalOccasion,
    build_context_membership,
    build_context_membership_design,
    build_longitudinal_design,
    build_longitudinal_state_spec,
    build_temporal_occasion,
)
from .estimation import (
    CrossedPersonEffectResult,
    estimate_crossed_person_effects,
    weighted_contextual_effect,
)

__all__ = [
    "ContextMembership",
    "ContextMembershipDesign",
    "LongitudinalDesign",
    "LongitudinalStateKind",
    "LongitudinalStateSpec",
    "MultilevelContractError",
    "TemporalOccasion",
    "build_context_membership",
    "build_context_membership_design",
    "build_longitudinal_design",
    "build_longitudinal_state_spec",
    "build_temporal_occasion",
    "CrossedPersonEffectResult",
    "estimate_crossed_person_effects",
    "weighted_contextual_effect",
]
