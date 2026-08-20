"""Contract tests for source-free cross-engine conformance manifests."""

from dataclasses import replace

import pytest

from fast_mlsirm.conformance import (
    ConformanceCapability,
    ConformanceLayer,
    ConformanceManifest,
    ConformanceProvenance,
    ConformanceStatus,
    ConformanceTolerance,
    EngineReference,
)


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_SHA_E = "e" * 64


def _engine(engine_id: str = "reference_engine") -> EngineReference:
    """Build one independent engine identity for the fixture."""
    return EngineReference(
        engine_id=engine_id,
        version="1.2.3",
        source_url="https://example.org/reference-engine",
        license_classification="open_source",
    )


def _tolerance(estimand_id: str = "response_probability") -> ConformanceTolerance:
    """Build one preregistered estimand tolerance."""
    return ConformanceTolerance(
        estimand_id=estimand_id,
        absolute=1e-8,
        relative=1e-6,
        rationale="double precision equation comparison",
    )


def _provenance() -> ConformanceProvenance:
    """Build reproducibility metadata bound to the protected revision."""
    return ConformanceProvenance(
        protected_main_sha=_SHA_A,
        harness_sha=_SHA_B,
        environment_fingerprint=_SHA_C,
        fixture_fingerprint=_SHA_D,
        mapping_schema="mapping_schema_v1",
        mapping_fingerprint=_SHA_E,
        rng_algorithm="pcg64",
        rng_seeds=(17, 23),
        raw_output_fingerprint=None,
        normalized_output_fingerprint=None,
        license_classification="synthetic_or_open",
    )


def _capability(
    capability_id: str = "rasch_probability",
    *,
    status: ConformanceStatus = ConformanceStatus.COVERED,
) -> ConformanceCapability:
    """Build one equation-level capability row."""
    return ConformanceCapability(
        capability_id=capability_id,
        public_entry_point="fast_mlsirm.irt_probability",
        estimand="item response probability under matched parameterization",
        layer=ConformanceLayer.EQUATION,
        status=status,
        engines=(_engine(),) if status is not ConformanceStatus.PLANNED else (),
        mapping_fingerprint=_SHA_E if status is ConformanceStatus.COVERED else None,
        tolerances=(_tolerance(),),
    )


def _manifest(*capabilities: ConformanceCapability) -> ConformanceManifest:
    """Build a bounded manifest fixture."""
    return ConformanceManifest(
        conformance_manifest_id="conformance_manifest_v1",
        protected_main_sha=_SHA_A,
        capabilities=capabilities,
        provenance=_provenance(),
    )


def test_manifest_is_deterministic_and_sorts_capabilities() -> None:
    """Capability order cannot change the content-addressed manifest identity."""
    first = _manifest(_capability("z_capability"), _capability("a_capability"))
    second = _manifest(_capability("a_capability"), _capability("z_capability"))

    assert first.manifest_fingerprint == second.manifest_fingerprint
    assert [row["capability_id"] for row in first.to_manifest()["capabilities"]] == [
        "a_capability",
        "z_capability",
    ]


def test_manifest_preserves_provider_neutral_engine_metadata() -> None:
    """The manifest records an engine reference without importing that engine."""
    manifest = _manifest(_capability())

    row = manifest.to_manifest()["capabilities"][0]
    assert row["engines"][0] == {
        "engine_id": "reference_engine",
        "license_classification": "open_source",
        "source_url": "https://example.org/reference-engine",
        "version": "1.2.3",
    }
    assert manifest.to_manifest()["provenance"]["rng_seeds"] == [17, 23]


def test_non_covered_capabilities_remain_explicit() -> None:
    """Planned coverage is represented without fabricating an external result."""
    capability = _capability("future_capability", status=ConformanceStatus.PLANNED)

    assert capability.engines == ()
    assert capability.mapping_fingerprint is None
    assert capability.status.value == "planned"


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        ("engine_id", "single", "lower snake_case"),
        ("source_url", "", "source_url must not be empty"),
        ("absolute", -1.0, "finite non-negative"),
        ("mapping_fingerprint", "bad", "lowercase SHA-256"),
    ],
)
def test_records_reject_invalid_public_inputs(
    field_name: str,
    value: object,
    match: str,
) -> None:
    """Malformed provenance and tolerance controls fail at construction."""
    if field_name in {"engine_id", "source_url"}:
        with pytest.raises(ValueError, match=match):
            replace(_engine(), **{field_name: value})
    elif field_name == "absolute":
        with pytest.raises(ValueError, match=match):
            replace(_tolerance(), absolute=value)
    else:
        with pytest.raises(ValueError, match=match):
            replace(_capability(), mapping_fingerprint=value)


def test_covered_capability_requires_mapping_and_engine() -> None:
    """A covered claim cannot omit its independent engine or mapping identity."""
    with pytest.raises(ValueError, match="independent engine"):
        ConformanceCapability(
            capability_id="covered_capability",
            public_entry_point="package.entry_point",
            estimand="matched estimand",
            layer=ConformanceLayer.EQUATION,
            status=ConformanceStatus.COVERED,
            engines=(),
            mapping_fingerprint=_SHA_E,
            tolerances=(_tolerance(),),
        )
    with pytest.raises(ValueError, match="mapping_fingerprint"):
        replace(_capability(), mapping_fingerprint=None)


def test_provenance_must_bind_the_same_protected_main_revision() -> None:
    """A manifest cannot combine a current code SHA with stale provenance."""
    provenance = replace(_provenance(), protected_main_sha=_SHA_B)

    with pytest.raises(ValueError, match="must match the manifest"):
        ConformanceManifest(
            conformance_manifest_id="conformance_manifest_v1",
            protected_main_sha=_SHA_A,
            capabilities=(_capability(),),
            provenance=provenance,
        )


def test_exact_package_records_reject_subclasses_before_field_callbacks() -> None:
    """Record admission rejects caller subclasses before reading their fields."""

    class HostileEngine(EngineReference):
        """Subclass whose field access must never be reached."""

        def __getattribute__(self, name: str) -> object:
            """Raise if record admission reads a subclass-controlled field."""
            if name == "engine_id":
                raise AssertionError("engine field callback executed")
            return super().__getattribute__(name)

    with pytest.raises(ValueError, match="exact package record"):
        HostileEngine(
            engine_id="reference_engine",
            version="1.2.3",
            source_url="https://example.org/reference-engine",
            license_classification="open_source",
        )
