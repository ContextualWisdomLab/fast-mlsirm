"""Factory-sealed public records for replay-verified pilot admission."""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from typing import Any
import weakref

from .audit import CandidateLifecycleState
from .audit import PILOT_RECORD_SCHEMA_VERSION
from .audit import PilotCandidateRecord as _CorePilotCandidateRecord

_PILOT_ADMISSION_TOKEN = object()
_PUBLIC_FIELD_NAMES = (
    "pilot_study_id",
    "query_testlet_id",
    "generator_family_id",
    "judge_policy_id",
    "occasion_id",
    "item_id",
    "candidate_fingerprint",
    "audit_report_fingerprint",
    "screening_result_fingerprint",
    "audit_policy_id",
    "audit_policy_version",
    "blueprint_id",
    "rubric_id",
    "rubric_version",
    "lifecycle_state",
    "schema_version",
)
_CREATION_SEALS: dict[
    int,
    tuple[weakref.ReferenceType[object], tuple[object, ...]],
] = {}


def _forget_creation_seal(
    record_key: int,
    reference: weakref.ReferenceType[object],
) -> None:
    """Discard one dead pilot-record seal without deleting a reused identity."""
    current = _CREATION_SEALS.get(record_key)
    if current is not None and current[0] is reference:
        _CREATION_SEALS.pop(record_key, None)


def _seal_record(record: PilotCandidateRecord) -> None:
    """Bind one admitted object identity to its normalized creation-time fields."""
    record_key = id(record)
    state = vars(record)
    snapshot = tuple(state[name] for name in _PUBLIC_FIELD_NAMES)
    reference = weakref.ref(
        record,
        lambda collected, key=record_key: _forget_creation_seal(key, collected),
    )
    _CREATION_SEALS[record_key] = (reference, snapshot)


def _verify_creation_seal(record: PilotCandidateRecord) -> dict[str, Any]:
    """Validate live fields and return the package-owned creation snapshot."""
    sealed = _CREATION_SEALS.get(id(record))
    if sealed is None or sealed[0]() is not record:
        raise ValueError("pilot record no longer matches its factory seal")
    state = vars(record)
    for name, expected in zip(_PUBLIC_FIELD_NAMES, sealed[1], strict=True):
        current = state.get(name)
        if type(current) is not type(expected) or current != expected:
            raise ValueError("pilot record no longer matches its factory seal")
    return dict(zip(_PUBLIC_FIELD_NAMES, sealed[1], strict=True))


@dataclass(frozen=True)
class PilotCandidateRecord:
    """Public pilot record created only after audit and screening replay succeeds.

    Python objects are not cryptographic capabilities. The private admission
    token prevents ordinary direct construction through the supported API, and
    a package-owned creation seal rejects later field rebinding before public
    serialization or identity replay. Downstream trust decisions must still
    verify the serialized fingerprints.
    """

    pilot_study_id: str
    query_testlet_id: str
    generator_family_id: str
    judge_policy_id: str
    occasion_id: str
    item_id: str
    candidate_fingerprint: str
    audit_report_fingerprint: str
    screening_result_fingerprint: str
    audit_policy_id: str
    audit_policy_version: str
    blueprint_id: str
    rubric_id: str
    rubric_version: str
    lifecycle_state: CandidateLifecycleState = CandidateLifecycleState.PILOT
    schema_version: str = PILOT_RECORD_SCHEMA_VERSION
    _admission_token: InitVar[object | None] = None

    def __post_init__(self, _admission_token: object | None) -> None:
        """Reject direct construction, normalize, and retain creation-time identity."""
        if _admission_token is not _PILOT_ADMISSION_TOKEN:
            raise ValueError(
                "PilotCandidateRecord must be created by "
                "build_pilot_candidate_record"
            )
        normalized = _CorePilotCandidateRecord(
            pilot_study_id=self.pilot_study_id,
            query_testlet_id=self.query_testlet_id,
            generator_family_id=self.generator_family_id,
            judge_policy_id=self.judge_policy_id,
            occasion_id=self.occasion_id,
            item_id=self.item_id,
            candidate_fingerprint=self.candidate_fingerprint,
            audit_report_fingerprint=self.audit_report_fingerprint,
            screening_result_fingerprint=self.screening_result_fingerprint,
            audit_policy_id=self.audit_policy_id,
            audit_policy_version=self.audit_policy_version,
            blueprint_id=self.blueprint_id,
            rubric_id=self.rubric_id,
            rubric_version=self.rubric_version,
            lifecycle_state=self.lifecycle_state,
            schema_version=self.schema_version,
        )
        for name in _PUBLIC_FIELD_NAMES:
            object.__setattr__(self, name, getattr(normalized, name))
        _seal_record(self)

    def _core_record(self) -> _CorePilotCandidateRecord:
        """Return the validated creation snapshot as an internal pilot record."""
        return _CorePilotCandidateRecord(**_verify_creation_seal(self))

    @property
    def pilot_record_fingerprint(self) -> str:
        """Return the complete SHA-256 identity of this pilot admission."""
        return self._core_record().pilot_record_fingerprint

    @property
    def pilot_record_id(self) -> str:
        """Return a descriptive 128-bit public handle for this pilot record."""
        return self._core_record().pilot_record_id

    def to_dict(self) -> dict[str, str]:
        """Return normalized pilot content plus deterministic identities."""
        return self._core_record().to_dict()


def _from_verified_core(
    record: _CorePilotCandidateRecord,
) -> PilotCandidateRecord:
    """Wrap a core record after the public policy has replay-verified admission."""
    if not isinstance(record, _CorePilotCandidateRecord):
        raise TypeError("record must be a core PilotCandidateRecord")
    return PilotCandidateRecord(
        pilot_study_id=record.pilot_study_id,
        query_testlet_id=record.query_testlet_id,
        generator_family_id=record.generator_family_id,
        judge_policy_id=record.judge_policy_id,
        occasion_id=record.occasion_id,
        item_id=record.item_id,
        candidate_fingerprint=record.candidate_fingerprint,
        audit_report_fingerprint=record.audit_report_fingerprint,
        screening_result_fingerprint=record.screening_result_fingerprint,
        audit_policy_id=record.audit_policy_id,
        audit_policy_version=record.audit_policy_version,
        blueprint_id=record.blueprint_id,
        rubric_id=record.rubric_id,
        rubric_version=record.rubric_version,
        lifecycle_state=record.lifecycle_state,
        schema_version=record.schema_version,
        _admission_token=_PILOT_ADMISSION_TOKEN,
    )
