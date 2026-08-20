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


def test_scalar_and_enum_boundaries_reject_invalid_values() -> None:
    """Cover non-text, oversized, non-enum, and unknown enum controls."""
    with pytest.raises(ValueError, match="version must be a string"):
        replace(_engine(), version=object())
    with pytest.raises(ValueError, match="at most 4096"):
        replace(_engine(), source_url="x" * 4_097)
    with pytest.raises(ValueError, match="supported ConformanceLayer"):
        replace(_capability(), layer=object())
    with pytest.raises(ValueError, match="layer must be one of"):
        replace(_capability(), layer="unknown")
    with pytest.raises(ValueError, match="finite non-negative"):
        replace(_tolerance(), absolute=object())
    with pytest.raises(ValueError, match="absolute or relative"):
        ConformanceTolerance(
            estimand_id="zero_tolerance",
            absolute=0.0,
            relative=0.0,
            rationale="invalid zero tolerance",
        )


def test_bounded_record_collections_reject_wrong_shape_size_and_type() -> None:
    """Bound nested records before accepting any caller-provided record."""
    with pytest.raises(ValueError, match="engines must be a list or tuple"):
        replace(_capability(), engines=object())
    too_many = tuple(_engine(f"engine_{index}") for index in range(65))
    with pytest.raises(ValueError, match="engines must contain at most 64"):
        replace(_capability(), engines=too_many)
    with pytest.raises(ValueError, match=r"engines\[0\] must be a EngineReference"):
        replace(_capability(), engines=(object(),))


def test_duplicate_nested_records_and_invalid_seeds_fail_closed() -> None:
    """Reject duplicate identities and malformed bounded RNG metadata."""
    with pytest.raises(ValueError, match="engine_id values must be unique"):
        replace(_capability(), engines=(_engine(), _engine()))
    with pytest.raises(ValueError, match="tolerance estimand_id values must be unique"):
        replace(_capability(), tolerances=(_tolerance(), _tolerance()))
    with pytest.raises(ValueError, match="rng_seeds must be a list or tuple"):
        replace(_provenance(), rng_seeds=object())
    with pytest.raises(ValueError, match="at most 64"):
        replace(_provenance(), rng_seeds=tuple(range(65)))
    with pytest.raises(ValueError, match="non-negative built-in integers"):
        replace(_provenance(), rng_seeds=(True,))


def test_exact_provenance_and_tolerance_records_reject_subclasses() -> None:
    """Nested package records cannot be replaced by caller-defined subclasses."""
    provenance = _provenance()
    tolerance = _tolerance()

    class HostileProvenance(ConformanceProvenance):
        """Subclass used to verify exact provenance admission."""

    class HostileTolerance(ConformanceTolerance):
        """Subclass used to verify exact tolerance admission."""

    with pytest.raises(ValueError, match="exact package record"):
        HostileProvenance(**{field: getattr(provenance, field) for field in provenance.__dataclass_fields__})
    with pytest.raises(ValueError, match="exact package record"):
        HostileTolerance(**{field: getattr(tolerance, field) for field in tolerance.__dataclass_fields__})


def test_manifest_boundary_rejects_wrong_provenance_empty_and_duplicate_capabilities() -> None:
    """Reject malformed manifest ownership, cardinality, and identity metadata."""
    with pytest.raises(ValueError, match="provenance must be a ConformanceProvenance"):
        replace(_manifest(_capability()), provenance=object())
    with pytest.raises(ValueError, match="capabilities must not be empty"):
        replace(_manifest(_capability()), capabilities=())
    with pytest.raises(ValueError, match="capability_id values must be unique"):
        _manifest(_capability("same_capability"), _capability("same_capability"))
    with pytest.raises(ValueError, match="schema_version must be '1.0'"):
        replace(_manifest(_capability()), schema_version="2.0")


def test_exact_capability_and_manifest_records_reject_subclasses() -> None:
    """Top-level package records reject subclass construction before normalization."""
    capability = _capability()
    manifest = _manifest(capability)

    class HostileCapability(ConformanceCapability):
        """Subclass used to verify exact capability admission."""

    class HostileManifest(ConformanceManifest):
        """Subclass used to verify exact manifest admission."""

    with pytest.raises(ValueError, match="exact package record"):
        HostileCapability(**{field: getattr(capability, field) for field in capability.__dataclass_fields__})
    with pytest.raises(ValueError, match="exact package record"):
        HostileManifest(**{field: getattr(manifest, field) for field in manifest.__dataclass_fields__})
