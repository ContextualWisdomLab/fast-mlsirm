"""Compatibility composition for public scoring contracts.

The implementation is split into bounded canonicalization, policy, assessment,
and observation modules. This file preserves one stable internal import surface
for the package namespace and future scoring components.
"""

from ._contract_safety import artifact_digest as artifact_digest
from ._contract_safety import canonical_json as canonical_json
from ._validation import ASSESSMENT_SCHEMA_VERSION as ASSESSMENT_SCHEMA_VERSION
from ._validation import (
    MAX_METADATA_COLLECTION_VALUES as MAX_METADATA_COLLECTION_VALUES,
)
from ._validation import MAX_METADATA_DEPTH as MAX_METADATA_DEPTH
from ._validation import MAX_METADATA_NODES as MAX_METADATA_NODES
from ._validation import AssessmentSpecError as AssessmentSpecError
from .assessment import AssessmentResponseType as AssessmentResponseType
from .assessment import AssessmentSpec as AssessmentSpec
from .assessment import ConstructSpec as ConstructSpec
from .assessment import build_assessment_spec as build_assessment_spec
from .observations import MAX_EVIDENCE_SPANS as MAX_EVIDENCE_SPANS
from .observations import OBSERVATION_SCHEMA_VERSION as OBSERVATION_SCHEMA_VERSION
from .observations import EngineDescriptor as EngineDescriptor
from .observations import EvidenceSourceKind as EvidenceSourceKind
from .observations import EvidenceSpan as EvidenceSpan
from .observations import ObservationStatus as ObservationStatus
from .observations import RaterKind as RaterKind
from .observations import ScoreObservation as ScoreObservation
from .observations import build_score_observation as build_score_observation
from .policies import AdjudicationPolicy as AdjudicationPolicy
from .policies import CalibrationPolicy as CalibrationPolicy
from .policies import EnginePolicy as EnginePolicy
from .policies import MonitoringPolicy as MonitoringPolicy
from .policies import ReportingPolicy as ReportingPolicy
from .policies import ValidationPolicy as ValidationPolicy

__all__ = [
    "ASSESSMENT_SCHEMA_VERSION",
    "OBSERVATION_SCHEMA_VERSION",
    "MAX_EVIDENCE_SPANS",
    "MAX_METADATA_COLLECTION_VALUES",
    "MAX_METADATA_DEPTH",
    "MAX_METADATA_NODES",
    "AdjudicationPolicy",
    "AssessmentResponseType",
    "AssessmentSpec",
    "AssessmentSpecError",
    "CalibrationPolicy",
    "ConstructSpec",
    "EngineDescriptor",
    "EnginePolicy",
    "EvidenceSourceKind",
    "EvidenceSpan",
    "MonitoringPolicy",
    "ObservationStatus",
    "RaterKind",
    "ReportingPolicy",
    "ScoreObservation",
    "ValidationPolicy",
    "artifact_digest",
    "build_assessment_spec",
    "build_score_observation",
    "canonical_json",
]
