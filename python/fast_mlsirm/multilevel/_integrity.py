"""Install fail-closed integrity guards for multilevel contract artifacts.

The public contracts are frozen dataclasses created only through package-owned
factories. Python's low-level :func:`object.__setattr__` can still mutate a
frozen instance, so every public identity and serialization surface verifies
its original content seal. The guards preserve exact-type checks at aggregate
factories and keep integrity policy isolated from numerical code.
"""

from __future__ import annotations

from collections.abc import Iterable
from types import ModuleType
from typing import Any, Callable


def install_contract_integrity(contracts: ModuleType) -> None:
    """Install idempotent exact-type and content-seal guards on ``contracts``."""
    if getattr(contracts, "_INTEGRITY_GUARDS_INSTALLED", False):
        return

    artifact_digest = contracts.artifact_digest
    contract_error = contracts.contract_error
    schema_version = contracts.MULTILEVEL_SCHEMA_VERSION

    ContextMembership = contracts.ContextMembership
    ContextMembershipDesign = contracts.ContextMembershipDesign
    TemporalOccasion = contracts.TemporalOccasion
    LongitudinalStateSpec = contracts.LongitudinalStateSpec
    LongitudinalDesign = contracts.LongitudinalDesign

    def assert_leaf(value: Any, *, code: str, message: str) -> None:
        """Verify one leaf artifact against the package-owned construction seal."""
        try:
            current = artifact_digest(value._content_dict())
        except Exception:
            current = ""
        if value.schema_version != schema_version or current != value._sealed_fingerprint:
            raise contract_error(code, "$", message)

    def membership_assert_integrity(self: Any) -> None:
        """Verify one contextual membership edge before exposing it."""
        assert_leaf(
            self,
            code="context_membership_integrity_mismatch",
            message="membership content no longer matches its package-owned seal",
        )

    def membership_fingerprint(self: Any) -> str:
        """Return the sealed membership identity after integrity verification."""
        membership_assert_integrity(self)
        return self._sealed_fingerprint

    def membership_handle(self: Any) -> str:
        """Return a descriptive 128-bit membership handle."""
        return f"context_membership_{membership_fingerprint(self)[:32]}"

    def membership_to_dict(self: Any) -> dict[str, Any]:
        """Return canonical membership content after seal verification."""
        membership_assert_integrity(self)
        return {
            **self._content_dict(),
            "membership_handle": membership_handle(self),
            "membership_fingerprint": membership_fingerprint(self),
        }

    ContextMembership._assert_integrity = membership_assert_integrity
    ContextMembership.membership_fingerprint = property(membership_fingerprint)
    ContextMembership.membership_handle = property(membership_handle)
    ContextMembership.to_dict = membership_to_dict

    def occasion_assert_integrity(self: Any) -> None:
        """Verify one temporal occasion before exposing it."""
        assert_leaf(
            self,
            code="temporal_occasion_integrity_mismatch",
            message="occasion content no longer matches its package-owned seal",
        )

    def occasion_fingerprint(self: Any) -> str:
        """Return the sealed occasion identity after integrity verification."""
        occasion_assert_integrity(self)
        return self._sealed_fingerprint

    def occasion_handle(self: Any) -> str:
        """Return a descriptive 128-bit occasion handle."""
        return f"temporal_occasion_{occasion_fingerprint(self)[:32]}"

    def occasion_to_dict(self: Any) -> dict[str, Any]:
        """Return canonical occasion content after seal verification."""
        occasion_assert_integrity(self)
        return {
            **self._content_dict(),
            "occasion_handle": occasion_handle(self),
            "occasion_fingerprint": occasion_fingerprint(self),
        }

    TemporalOccasion._assert_integrity = occasion_assert_integrity
    TemporalOccasion.occasion_fingerprint = property(occasion_fingerprint)
    TemporalOccasion.occasion_handle = property(occasion_handle)
    TemporalOccasion.to_dict = occasion_to_dict

    def state_assert_integrity(self: Any) -> None:
        """Verify one longitudinal state specification before exposing it."""
        assert_leaf(
            self,
            code="longitudinal_state_spec_integrity_mismatch",
            message="state specification no longer matches its package-owned seal",
        )

    def state_fingerprint(self: Any) -> str:
        """Return the sealed state identity after integrity verification."""
        state_assert_integrity(self)
        return self._sealed_fingerprint

    def state_handle(self: Any) -> str:
        """Return a descriptive 128-bit state-specification handle."""
        return f"longitudinal_state_spec_{state_fingerprint(self)[:32]}"

    def state_to_dict(self: Any) -> dict[str, Any]:
        """Return canonical state content after seal verification."""
        state_assert_integrity(self)
        return {
            **self._content_dict(),
            "state_spec_handle": state_handle(self),
            "state_spec_fingerprint": state_fingerprint(self),
        }

    LongitudinalStateSpec._assert_integrity = state_assert_integrity
    LongitudinalStateSpec.state_spec_fingerprint = property(state_fingerprint)
    LongitudinalStateSpec.state_spec_handle = property(state_handle)
    LongitudinalStateSpec.to_dict = state_to_dict

    original_membership_design_init = ContextMembershipDesign.__post_init__
    original_longitudinal_design_init = LongitudinalDesign.__post_init__

    def membership_design_init(self: Any, token: object | None) -> None:
        """Preserve factory validation and seal complete aggregate content."""
        original_membership_design_init(self, token)
        object.__setattr__(self, "_sealed_fingerprint", artifact_digest(self._content_dict()))

    def longitudinal_design_init(self: Any, token: object | None) -> None:
        """Preserve factory validation and seal complete longitudinal content."""
        original_longitudinal_design_init(self, token)
        object.__setattr__(self, "_sealed_fingerprint", artifact_digest(self._content_dict()))

    ContextMembershipDesign.__post_init__ = membership_design_init
    LongitudinalDesign.__post_init__ = longitudinal_design_init

    def assert_membership_design(self: Any) -> None:
        """Verify the aggregate seal and every exact child artifact."""
        try:
            if any(type(value) is not ContextMembership for value in self.memberships):
                raise TypeError
            for value in self.memberships:
                membership_assert_integrity(value)
            current = artifact_digest(self._content_dict())
        except Exception:
            current = ""
        if (
            self.schema_version != schema_version
            or current != getattr(self, "_sealed_fingerprint", None)
        ):
            raise contract_error(
                "context_membership_design_integrity_mismatch",
                "$",
                "membership design no longer matches its package-owned seal",
            )

    def membership_design_fingerprint(self: Any) -> str:
        """Return the sealed aggregate identity after integrity verification."""
        assert_membership_design(self)
        return self._sealed_fingerprint

    def membership_design_handle(self: Any) -> str:
        """Return a descriptive 128-bit membership-design handle."""
        return f"context_membership_design_{membership_design_fingerprint(self)[:32]}"

    def conditional_context_ids(self: Any) -> tuple[str, ...]:
        """Expose legacy labels only for one unambiguous context dimension."""
        if len(self.context_dimension_ids) != 1:
            raise AttributeError(
                "context_ids is ambiguous for multi-dimensional context designs; use context_keys"
            )
        return tuple(sorted({value.context_id for value in self.memberships}))

    def membership_design_to_dict(self: Any) -> dict[str, Any]:
        """Return aggregate content without ambiguous unqualified context labels."""
        assert_membership_design(self)
        return {
            "schema_version": self.schema_version,
            "memberships": [value.to_dict() for value in self.memberships],
            "observation_ids": list(self.observation_ids),
            "context_dimension_ids": list(self.context_dimension_ids),
            "context_keys": [list(value) for value in self.context_keys],
            "membership_counts": list(self.membership_counts),
            "membership_weights": [list(value) for value in self.membership_weights],
            "membership_counts_by_dimension": [
                list(value) for value in self.membership_counts_by_dimension
            ],
            "membership_weights_by_dimension": [
                [list(weights) for weights in observation]
                for observation in self.membership_weights_by_dimension
            ],
            "design_handle": membership_design_handle(self),
            "design_fingerprint": membership_design_fingerprint(self),
        }

    ContextMembershipDesign._assert_integrity = assert_membership_design
    ContextMembershipDesign.design_fingerprint = property(membership_design_fingerprint)
    ContextMembershipDesign.design_handle = property(membership_design_handle)
    ContextMembershipDesign.context_ids = property(conditional_context_ids)
    ContextMembershipDesign.to_dict = membership_design_to_dict

    def assert_longitudinal_design(self: Any) -> None:
        """Verify the longitudinal aggregate seal and every exact child artifact."""
        try:
            if type(self.state_spec) is not LongitudinalStateSpec:
                raise TypeError
            if any(type(value) is not TemporalOccasion for value in self.occasions):
                raise TypeError
            state_assert_integrity(self.state_spec)
            for value in self.occasions:
                occasion_assert_integrity(value)
            current = artifact_digest(self._content_dict())
        except Exception:
            current = ""
        if (
            self.schema_version != schema_version
            or current != getattr(self, "_sealed_fingerprint", None)
        ):
            raise contract_error(
                "longitudinal_design_integrity_mismatch",
                "$",
                "longitudinal design no longer matches its package-owned seal",
            )

    def longitudinal_design_fingerprint(self: Any) -> str:
        """Return the sealed longitudinal identity after integrity verification."""
        assert_longitudinal_design(self)
        return self._sealed_fingerprint

    def longitudinal_design_handle(self: Any) -> str:
        """Return a descriptive 128-bit longitudinal-design handle."""
        return f"longitudinal_design_{longitudinal_design_fingerprint(self)[:32]}"

    def longitudinal_design_to_dict(self: Any) -> dict[str, Any]:
        """Return longitudinal content after aggregate integrity verification."""
        assert_longitudinal_design(self)
        return {
            "schema_version": self.schema_version,
            "occasions": [value.to_dict() for value in self.occasions],
            "state_spec": self.state_spec.to_dict(),
            "respondent_ids": list(self.respondent_ids),
            "occasion_counts": list(self.occasion_counts),
            "design_handle": longitudinal_design_handle(self),
            "design_fingerprint": longitudinal_design_fingerprint(self),
        }

    LongitudinalDesign._assert_integrity = assert_longitudinal_design
    LongitudinalDesign.design_fingerprint = property(longitudinal_design_fingerprint)
    LongitudinalDesign.design_handle = property(longitudinal_design_handle)
    LongitudinalDesign.to_dict = longitudinal_design_to_dict

    original_membership_builder: Callable[..., Any] = contracts.build_context_membership_design
    original_longitudinal_builder: Callable[..., Any] = contracts.build_longitudinal_design

    def exact_membership_builder(memberships: Iterable[Any]) -> Any:
        """Build a design only from exact package-owned membership values."""
        raw = contracts._safe_values(
            memberships,
            "memberships",
            minimum=1,
            maximum=contracts.MAX_CONTEXT_MEMBERSHIPS,
        )
        for index, value in enumerate(raw):
            if type(value) is not ContextMembership:
                raise contract_error(
                    "invalid_context_membership",
                    f"$.memberships[{index}]",
                    "memberships must contain exact ContextMembership values",
                )
        return original_membership_builder(raw)

    def exact_longitudinal_builder(
        *, occasions: Iterable[Any], state_spec: Any
    ) -> Any:
        """Build a design only from exact package-owned occasion and state values."""
        if type(state_spec) is not LongitudinalStateSpec:
            raise contract_error(
                "invalid_longitudinal_state_spec",
                "$.state_spec",
                "state_spec must be an exact LongitudinalStateSpec",
            )
        raw = contracts._safe_values(
            occasions,
            "occasions",
            minimum=1,
            maximum=contracts.MAX_TEMPORAL_OCCASIONS,
        )
        for index, value in enumerate(raw):
            if type(value) is not TemporalOccasion:
                raise contract_error(
                    "invalid_temporal_occasion",
                    f"$.occasions[{index}]",
                    "occasions must contain exact TemporalOccasion values",
                )
        return original_longitudinal_builder(occasions=raw, state_spec=state_spec)

    contracts.build_context_membership_design = exact_membership_builder
    contracts.build_longitudinal_design = exact_longitudinal_builder
    contracts._INTEGRITY_GUARDS_INSTALLED = True
