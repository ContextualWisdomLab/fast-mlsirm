"""Persisted-manifest replay regressions for cross-engine conformance."""

from __future__ import annotations

import json

import pytest

from fast_mlsirm.cross_engine_conformance import (
    ConformanceCapability,
    ConformanceCoverageStatus,
    ConformanceInventory,
    ConformanceRunProvenance,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_SOURCE_COMMIT = "0" * 40


class _HostileDict(dict[str, object]):
    """Dictionary subtype that records unsafe replay callback dispatch."""

    callbacks = 0

    def __iter__(self):  # type: ignore[override]
        """Reject iteration before exact built-in mapping admission."""
        type(self).callbacks += 1
        raise AssertionError("hostile dictionary iteration executed")

    def __getitem__(self, key: str) -> object:
        """Reject lookup before exact built-in mapping admission."""
        type(self).callbacks += 1
        raise AssertionError("hostile dictionary lookup executed")


class _HostileList(list[object]):
    """List subtype that records unsafe replay callback dispatch."""

    callbacks = 0

    def __iter__(self):  # type: ignore[override]
        """Reject iteration before exact built-in list admission."""
        type(self).callbacks += 1
        raise AssertionError("hostile list iteration executed")


class _HostileText(str):
    """String subtype that records unsafe JSON-input callback dispatch."""

    callbacks = 0

    def __len__(self) -> int:
        """Reject length access before exact built-in text admission."""
        type(self).callbacks += 1
        raise AssertionError("hostile string length executed")


def _provenance() -> ConformanceRunProvenance:
    """Return one canonical source-free run provenance record."""
    return ConformanceRunProvenance(
        harness_commit=_SOURCE_COMMIT,
        environment_sha256=_SHA_A,
        environment_kind="environment_lock",
        operating_system="linux",
        architecture="x86_64",
        rng_algorithm="pcg64_dxsm",
        rng_seeds=(17, 23),
        mapping_schema_version="1.0.0",
        mapping_sha256=_SHA_B,
        model_configuration_sha256=_SHA_C,
        convergence_controls_sha256=_SHA_D,
        tolerance_sha256=_SHA_A,
        tolerance_rationale="fixed-parameter double-precision comparison",
        raw_output_sha256=None,
        normalized_output_sha256=None,
        license_classification="synthetic_or_open",
        redistribution_status="metadata_only",
    )


def _inventory() -> ConformanceInventory:
    """Return one canonical nonexecuted inventory for replay tests."""
    capability = ConformanceCapability(
        capability_id="rasch_probability",
        public_entrypoint="fast_mlsirm.rasch.probability",
        estimand="Dichotomous Rasch response probability",
        likelihood_family="bernoulli_logit",
        parameterization="difficulty with unit discrimination",
        identification="latent location fixed by the compared fixture",
        comparison_scope="fixed-parameter equation conformance",
        coverage_status=ConformanceCoverageStatus.PLANNED,
        evidence=(),
    )
    return ConformanceInventory(
        package_version="0.8.0",
        source_commit=_SOURCE_COMMIT,
        capabilities=(capability,),
        run_provenance=_provenance(),
    )


def _canonical_json(payload: object) -> str:
    """Return the same deterministic JSON shape emitted by conformance tooling."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def test_manifest_round_trip_revalidates_exact_inventory_identity() -> None:
    """A canonical persisted manifest rehydrates through package-owned validation."""
    inventory = _inventory()
    manifest = inventory.to_manifest()

    replayed = ConformanceInventory.from_manifest(manifest)

    assert replayed == inventory
    assert replayed.inventory_fingerprint == inventory.inventory_fingerprint
    assert replayed.to_manifest() == manifest


def test_json_round_trip_revalidates_exact_inventory_identity() -> None:
    """Canonical JSON replays to the same inventory and manifest fingerprint."""
    inventory = _inventory()
    payload = _canonical_json(inventory.to_manifest())

    replayed = ConformanceInventory.from_json(payload)

    assert replayed == inventory
    assert replayed.to_manifest() == inventory.to_manifest()


def test_manifest_replay_rejects_tampered_fingerprint() -> None:
    """A stored fingerprint must match the canonical revalidated payload."""
    manifest = _inventory().to_manifest()
    manifest["inventory_fingerprint"] = "f" * 64

    with pytest.raises(ValueError, match="inventory_fingerprint does not match"):
        ConformanceInventory.from_manifest(manifest)


@pytest.mark.parametrize("mutation", ["unknown", "missing"])
def test_manifest_replay_requires_exact_root_key_set(mutation: str) -> None:
    """Unknown and missing root fields fail closed instead of being ignored."""
    manifest = _inventory().to_manifest()
    if mutation == "unknown":
        manifest["unexpected"] = "value"
    else:
        del manifest["source_commit"]

    with pytest.raises(ValueError, match="manifest keys must be exactly"):
        ConformanceInventory.from_manifest(manifest)


def test_manifest_replay_requires_exact_nested_key_sets() -> None:
    """Unknown nested capability fields cannot survive replay validation."""
    manifest = _inventory().to_manifest()
    capabilities = manifest["capabilities"]
    assert type(capabilities) is list
    capability = capabilities[0]
    assert type(capability) is dict
    capability["unexpected"] = "value"

    with pytest.raises(ValueError, match="capability manifest keys must be exactly"):
        ConformanceInventory.from_manifest(manifest)


def test_manifest_replay_rejects_hostile_mapping_without_callbacks() -> None:
    """Caller-defined mapping subtypes are rejected before iteration or lookup."""
    _HostileDict.callbacks = 0
    hostile = _HostileDict(_inventory().to_manifest())

    with pytest.raises(ValueError, match="manifest must be a built-in dictionary"):
        ConformanceInventory.from_manifest(hostile)

    assert _HostileDict.callbacks == 0


def test_manifest_replay_rejects_hostile_nested_list_without_callbacks() -> None:
    """Caller-defined list subtypes are rejected before capability iteration."""
    _HostileList.callbacks = 0
    manifest = _inventory().to_manifest()
    manifest["capabilities"] = _HostileList(manifest["capabilities"])

    with pytest.raises(ValueError, match="capabilities must be a built-in list"):
        ConformanceInventory.from_manifest(manifest)

    assert _HostileList.callbacks == 0


def test_json_replay_rejects_duplicate_object_keys() -> None:
    """Duplicate JSON keys cannot be collapsed before provenance validation."""
    payload = '{"schema_version":"1.0","schema_version":"1.0"}'

    with pytest.raises(ValueError, match="duplicate JSON object key: schema_version"):
        ConformanceInventory.from_json(payload)


def test_json_replay_rejects_oversized_payload_before_parsing() -> None:
    """Persisted manifest parsing has a fixed pre-parse resource bound."""
    payload = " " * 1_048_577

    with pytest.raises(ValueError, match="manifest JSON must contain at most 1048576"):
        ConformanceInventory.from_json(payload)


def test_json_replay_rejects_text_subclass_before_callbacks() -> None:
    """Caller-defined JSON text subtypes are rejected before length or parsing."""
    _HostileText.callbacks = 0

    with pytest.raises(ValueError, match="manifest JSON must be a string"):
        ConformanceInventory.from_json(_HostileText("{}"))

    assert _HostileText.callbacks == 0


def test_manifest_replay_rejects_noncanonical_normalized_text() -> None:
    """Replay cannot silently normalize a persisted signed manifest in place."""
    manifest = _inventory().to_manifest()
    provenance = manifest["run_provenance"]
    assert type(provenance) is dict
    provenance["operating_system"] = " linux "

    with pytest.raises(ValueError, match="manifest must already be canonical"):
        ConformanceInventory.from_manifest(manifest)
