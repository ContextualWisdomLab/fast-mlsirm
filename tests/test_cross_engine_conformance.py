"""Behavioral contract for cross-engine numerical conformance inventory."""

from __future__ import annotations

import json
import re

import pytest

from fast_mlsirm.cross_engine_conformance import (
    ComparisonEngine,
    ConformanceCapability,
    ConformanceCoverageStatus,
    ConformanceEvidence,
    ConformanceExecutionStatus,
    ConformanceInventory,
    ConformanceLayer,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_SOURCE_COMMIT = "0" * 40


class _HostileText(str):
    """String subclass whose text callbacks expose unsafe admission."""

    callbacks = 0

    def strip(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        """Record unsafe string normalization."""
        type(self).callbacks += 1
        raise AssertionError("hostile strip callback executed")

    def encode(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        """Record unsafe byte conversion."""
        type(self).callbacks += 1
        raise AssertionError("hostile encode callback executed")


class _HostileList(list[object]):
    """List subclass whose iteration exposes unsafe collection admission."""

    callbacks = 0

    def __iter__(self):  # type: ignore[override]
        """Record unsafe iteration."""
        type(self).callbacks += 1
        raise AssertionError("hostile iteration executed")


def _engine(*, engine_id: str = "mirt_engine") -> ComparisonEngine:
    """Return one isolated external-engine identity fixture."""
    return ComparisonEngine(
        engine_id=engine_id,
        engine_version="1.44.0",
        source_reference="doi:10.18637/jss.v048.i06",
        license_classification="gpl_3_reviewed",
    )


def _evidence(
    *,
    evidence_id: str = "rasch_probability_check",
    engine: ComparisonEngine | None = None,
    layer: ConformanceLayer = ConformanceLayer.FIXED_PARAMETER_EQUATION,
    execution_status: ConformanceExecutionStatus = ConformanceExecutionStatus.PASSED,
    artifact_sha256: str | None = _SHA_D,
) -> ConformanceEvidence:
    """Return one versioned comparison-evidence fixture."""
    return ConformanceEvidence(
        evidence_id=evidence_id,
        engine=engine or _engine(),
        layer=layer,
        execution_status=execution_status,
        parameter_mapping_version="1.0.0",
        parameter_mapping_sha256=_SHA_A,
        fixture_sha256=_SHA_B,
        environment_sha256=_SHA_C,
        artifact_sha256=artifact_sha256,
        limitation=None,
    )


def _capability(
    *,
    capability_id: str = "rasch_probability",
    evidence: tuple[ConformanceEvidence, ...] | list[ConformanceEvidence] | None = None,
    coverage_status: ConformanceCoverageStatus = ConformanceCoverageStatus.COVERED,
) -> ConformanceCapability:
    """Return one public-capability conformance fixture."""
    rows = (_evidence(),) if evidence is None else evidence
    return ConformanceCapability(
        capability_id=capability_id,
        public_entrypoint="fast_mlsirm.rasch.probability",
        estimand="Dichotomous Rasch response probability",
        likelihood_family="bernoulli_logit",
        parameterization="difficulty with unit discrimination",
        identification="latent location fixed by the compared fixture",
        comparison_scope="fixed-parameter equation conformance",
        coverage_status=coverage_status,
        evidence=rows,
    )


def test_inventory_is_deterministic_source_free_and_json_serializable() -> None:
    """Normalized capability order yields one content-addressed manifest."""
    first_capability = _capability(capability_id="rasch_probability")
    second_capability = _capability(
        capability_id="rasch_log_likelihood",
        evidence=(
            _evidence(
                evidence_id="rasch_log_likelihood_check",
                layer=ConformanceLayer.FIXED_PARAMETER_EQUATION,
            ),
        ),
    )
    first = ConformanceInventory(
        package_version="0.8.0",
        source_commit=_SOURCE_COMMIT,
        capabilities=[first_capability, second_capability],
    )
    second = ConformanceInventory(
        package_version="0.8.0",
        source_commit=_SOURCE_COMMIT,
        capabilities=(second_capability, first_capability),
    )

    assert first == second
    assert first.inventory_fingerprint == second.inventory_fingerprint
    assert re.fullmatch(r"[0-9a-f]{64}", first.inventory_fingerprint)
    manifest = first.to_manifest()
    assert [row["capability_id"] for row in manifest["capabilities"]] == [
        "rasch_log_likelihood",
        "rasch_probability",
    ]
    assert "raw_response" not in json.dumps(manifest)
    assert json.loads(json.dumps(manifest, ensure_ascii=False)) == manifest


def test_not_executed_remains_distinct_from_passed() -> None:
    """An unavailable optional engine cannot silently become passing evidence."""
    not_executed = _evidence(
        execution_status=ConformanceExecutionStatus.NOT_EXECUTED,
        artifact_sha256=None,
    )
    passed = _evidence()

    assert not_executed.execution_status is ConformanceExecutionStatus.NOT_EXECUTED
    assert not_executed.to_manifest()["execution_status"] == "not_executed"
    assert passed.to_manifest()["execution_status"] == "passed"
    assert not_executed.to_manifest() != passed.to_manifest()


@pytest.mark.parametrize(
    "status",
    [
        ConformanceExecutionStatus.PASSED,
        ConformanceExecutionStatus.FAILED,
        ConformanceExecutionStatus.INDETERMINATE,
    ],
)
def test_executed_statuses_require_an_artifact(status: ConformanceExecutionStatus) -> None:
    """Executed verdicts require immutable result evidence."""
    with pytest.raises(ValueError, match="artifact_sha256 is required"):
        _evidence(execution_status=status, artifact_sha256=None)


@pytest.mark.parametrize(
    "status",
    [
        ConformanceExecutionStatus.NOT_EXECUTED,
        ConformanceExecutionStatus.NOT_APPLICABLE,
    ],
)
def test_nonexecuted_statuses_reject_result_artifacts(
    status: ConformanceExecutionStatus,
) -> None:
    """Nonexecuted verdicts cannot carry a result artifact that implies a run."""
    with pytest.raises(ValueError, match="artifact_sha256 must be omitted"):
        _evidence(execution_status=status, artifact_sha256=_SHA_D)


def test_coverage_states_fail_closed_on_contradictory_evidence() -> None:
    """Coverage metadata cannot claim support without a compatible evidence set."""
    with pytest.raises(ValueError, match="covered capability requires evidence"):
        _capability(evidence=(), coverage_status=ConformanceCoverageStatus.COVERED)

    with pytest.raises(ValueError, match="must not contain comparison evidence"):
        _capability(
            coverage_status=ConformanceCoverageStatus.NO_INDEPENDENT_ENGINE,
            evidence=(_evidence(),),
        )

    planned = _capability(
        coverage_status=ConformanceCoverageStatus.PLANNED,
        evidence=(),
    )
    assert planned.coverage_status is ConformanceCoverageStatus.PLANNED


def test_duplicate_capabilities_and_evidence_are_rejected() -> None:
    """Inventory identities remain unique at both hierarchy levels."""
    duplicate_evidence = (_evidence(), _evidence())
    with pytest.raises(ValueError, match="evidence_id values must be unique"):
        _capability(evidence=duplicate_evidence)

    capability = _capability()
    with pytest.raises(ValueError, match="capability_id values must be unique"):
        ConformanceInventory(
            package_version="0.8.0",
            source_commit=_SOURCE_COMMIT,
            capabilities=(capability, capability),
        )


def test_hostile_text_subclasses_are_rejected_without_callbacks() -> None:
    """Semantic text is sealed before strip, hashing, or digest work."""
    _HostileText.callbacks = 0
    hostile = _HostileText("1.44.0")

    with pytest.raises(ValueError, match="engine_version must be a string"):
        ComparisonEngine(
            engine_id="mirt_engine",
            engine_version=hostile,
            source_reference="doi:10.18637/jss.v048.i06",
            license_classification="gpl_3_reviewed",
        )

    assert _HostileText.callbacks == 0


def test_hostile_collection_subclass_is_rejected_without_iteration() -> None:
    """Capability collections require inert built-in containers before iteration."""
    _HostileList.callbacks = 0
    hostile = _HostileList([_capability()])

    with pytest.raises(ValueError, match="capabilities must be a list or tuple"):
        ConformanceInventory(
            package_version="0.8.0",
            source_commit=_SOURCE_COMMIT,
            capabilities=hostile,
        )

    assert _HostileList.callbacks == 0


def test_mutated_nested_engine_is_revalidated_before_text_callbacks() -> None:
    """Nested package records are reconstructed from callback-free scalar identity."""
    _HostileText.callbacks = 0
    engine = _engine()
    object.__setattr__(engine, "engine_version", _HostileText("1.44.0"))

    with pytest.raises(ValueError, match="engine_version must be a string"):
        _evidence(engine=engine)

    assert _HostileText.callbacks == 0


def test_fingerprint_changes_when_evidence_verdict_changes() -> None:
    """Execution verdicts participate in immutable inventory provenance."""
    passed = ConformanceInventory(
        package_version="0.8.0",
        source_commit=_SOURCE_COMMIT,
        capabilities=(_capability(),),
    )
    failed = ConformanceInventory(
        package_version="0.8.0",
        source_commit=_SOURCE_COMMIT,
        capabilities=(
            _capability(
                evidence=(
                    _evidence(
                        execution_status=ConformanceExecutionStatus.FAILED,
                    ),
                ),
            ),
        ),
    )

    assert passed.inventory_fingerprint != failed.inventory_fingerprint
